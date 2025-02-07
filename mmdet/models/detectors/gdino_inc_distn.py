# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import os
import json
import copy
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import torch
import torch.nn as nn
from torch import Tensor
from mmengine.runner import load_checkpoint, load_state_dict
from mmengine.model import is_model_wrapper
from mmengine import Config
from mmengine.structures import InstanceData
from mmdet.utils import ConfigType, OptConfigType, InstanceList
from mmdet.structures.bbox import bbox2roi
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, bbox_overlaps
from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores
from mmdet.structures import OptSampleList, SampleList
from ..layers import SinePositionalEncoding, CdnQueryGenerator
from ..layers import inverse_sigmoid
from .gdino_inc import GroundingDINO_inc

@MODELS.register_module()
class GroundingDINO_inc_distn(GroundingDINO_inc):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self, distn_cfg, dn_cfg: OptConfigType = None, vis_cfg: OptConfigType = None, *args, **kwargs) -> None:
        self.distn_cfg = distn_cfg
        if 'feat_distn' not in self.distn_cfg:
            self.distn_cfg.feat_distn = Config._dict_to_config_dict_lazy(dict(type='None'))
        if 'query_distn' not in self.distn_cfg:
            self.distn_cfg.query_distn = Config._dict_to_config_dict_lazy(dict(type='None'))
        super().__init__(*args, **kwargs)
        self.start=self.bbox_head.trunc_class[0]
        self.end=self.bbox_head.trunc_class[1]
        self.dn_cfg=dn_cfg 
        self.max_text_len=self.language_model.max_tokens
        self.token_positive_maps=None
        self.ori_token_positive_maps=None
        self.new_text_token_mask_chunked=None

        if self.dn_cfg is not None:
            self.dn_cfg['num_classes'] = self.bbox_head.trunc_class[1] 
            self.dn_cfg['embed_dims'] = self.embed_dims
            self.dn_cfg['num_matching_queries'] = self.num_queries
            self.dn_query_generator = CdnQueryGenerator(**self.dn_cfg)
        self.load_base_detector()

    def init_weights(self) -> None:
        pass

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(GroundingDINO_inc, self).train(mode)
        self._freeze_stages_train()
        self.ori_model.eval()

    def load_base_detector(self):
        ori_cfg = Config.fromfile(self.distn_cfg['ori_config_file'])
        ori_model = MODELS.build(ori_cfg.model)
        ori_model.eval()
        for param in ori_model.parameters():
            param.requires_grad = False
        self.ori_model = ori_model
        
    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
        aux_dict: Dict = None
    ) -> Dict:

        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)
        
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, aux_dict=aux_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        return head_inputs_dict
    
    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        aux_dict: Dict = None,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training :
            if self.dn_cfg is not None:
                dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(batch_data_samples)
                query = torch.cat([dn_label_query, query], dim=1)
                reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)
            else:
                dn_mask, dn_meta = None, None                
                reference_points = topk_coords_unact
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None

        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        if self.training:
            head_inputs_dict = dict(enc_outputs_class=topk_score, enc_outputs_coord=topk_coords, 
                                    dn_meta=dn_meta) 
        else:
            head_inputs_dict = dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict  
      
    def forward_ori_model(
            self,
            img_feats: Tuple[Tensor],
            text_dict: Dict,
            batch_data_samples: OptSampleList = None,
        ):
        encoder_inputs_dict, decoder_inputs_dict = self.ori_model.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.ori_model.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.ori_model.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples, query_distn=self.distn_cfg.query_distn)
        decoder_inputs_dict.update(tmp_dec_in)

        if self.distn_cfg.query_distn.type == 'seperate_queryinit':
            head_inputs_dict['aux_query'] = tmp_dec_in['query'].clone()
            head_inputs_dict['aux_reference'] = tmp_dec_in['reference_points'].clone()

        decoder_outputs_dict = self.ori_model.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)      

        head_inputs_dict['ori_text_token_mask'] = head_inputs_dict.pop('text_token_mask')
        head_inputs_dict['ori_memory_text'] = head_inputs_dict.pop('memory_text')
        head_inputs_dict['ori_hidden_states'] = head_inputs_dict.pop('hidden_states')
        head_inputs_dict['ori_references'] = head_inputs_dict.pop('references')

        return head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:

        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]
        # text for ori model 
        ori_text_prompts = [
            data_samples.ori_text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]
        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []

            if len(set(text_prompts)) == 1:
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)

                new_text_prompts = [caption_string] * len(batch_inputs)

                ori_tokenized, ori_caption_string, ori_tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        ori_text_prompts[0], True)
                ori_text_prompts = [ori_caption_string] * len(batch_inputs)

                if self.token_positive_maps is None:
                    token_positive_maps, _ = self.get_positive_map(tokenized, tokens_positive)
                    self.token_positive_maps = token_positive_maps
                    ori_token_positive_maps, _ = self.get_positive_map(ori_tokenized, ori_tokens_positive)
                    self.ori_token_positive_maps = ori_token_positive_maps   

                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    # NOTE construct a map such that positive_map[i,j] = True if box i is associated to token j
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        # new text forward
        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])   # [in_feat = 768, out_feat = 256]

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
        
        # chunked new class text token mask for full class prompt
        if self.new_text_token_mask_chunked is None:
            new_text_token_mask_chunk_pos = self.token_positive_maps[self.start+1][0]
            new_text_token_mask_chunked = copy.deepcopy(text_token_mask)  
            new_text_token_mask_chunked[:new_text_token_mask_chunk_pos] = False
            new_text_token_mask_chunked = new_text_token_mask_chunked.unsqueeze(0).repeat(len(batch_data_samples),1)
            
        # ori text forward（given same text as new model but trunc accroding to ori_text_token_mask）
        with torch.no_grad():
            # ori model forward on full text
            if self.distn_cfg.future_class:
                ori_text_dict = self.ori_model.language_model(new_text_prompts)
            else:
                ori_text_dict = self.ori_model.language_model(ori_text_prompts)

            ori_text_dict['embedded'] = self.ori_model.text_feat_map(ori_text_dict['embedded'])
            ori_visual_features = self.ori_model.extract_feat(batch_inputs)
            ori_head_inputs_dict = self.forward_ori_model(ori_visual_features, ori_text_dict, batch_data_samples)
            all_layers_ori_cls_scores, all_layers_ori_bbox_preds = \
                self.ori_model.bbox_head(ori_head_inputs_dict['ori_hidden_states'], 
                                        ori_head_inputs_dict['ori_references'], 
                                        ori_head_inputs_dict['ori_memory_text'], 
                                        ori_head_inputs_dict['ori_text_token_mask'])
            ori_head_inputs_dict['all_layers_ori_cls_scores'] = all_layers_ori_cls_scores
            ori_head_inputs_dict['all_layers_ori_bbox_preds'] = all_layers_ori_bbox_preds
            ori_head_inputs_dict['ori_token_positive_maps'] = self.ori_token_positive_maps
        
            if self.distn_cfg.future_class:
                ori_text_len = self.ori_token_positive_maps[len(data_samples.ori_text)][-1] + 1
                ori_text_token_mask =  ori_head_inputs_dict['ori_text_token_mask'][:,:ori_text_len]
                ori_head_inputs_dict['ori_text_token_mask'] = ori_text_token_mask
            else:
                ori_text_token_mask = ori_head_inputs_dict['ori_text_token_mask']

            if self.distn_cfg.label_distn.type == 'topk_pseudo' or self.distn_cfg.label_distn.type == 'threshold_pseudo':
                topk_query, batch_pseudo_instances, batch_all_instances = \
                    self.bbox_head.generate_pseudo_label(all_layers_ori_cls_scores,
                                                        all_layers_ori_bbox_preds,
                                                        ori_text_token_mask,
                                                        text_token_mask, 
                                                        batch_data_samples,
                                                        self.ori_token_positive_maps)    
            else:
                topk_query = None
                batch_pseudo_instances = None
                batch_all_instances = None
            
            ori_head_inputs_dict['batch_pseudo_instances'] = batch_pseudo_instances
            ori_head_inputs_dict['batch_all_instances'] = batch_all_instances
            ori_head_inputs_dict['ori_topk_query'] = topk_query

        # new model forward, 
        visual_features = self.extract_feat(batch_inputs)

        aux_dict = None
        if self.distn_cfg.query_distn.type == 'seperate_queryinit':
            aux_query = ori_head_inputs_dict['aux_query']
            aux_enc_coord = ori_head_inputs_dict['enc_outputs_coord'] 
            aux_enc_score = ori_head_inputs_dict['enc_outputs_class']
            aux_dict = dict(aux_query=aux_query, aux_enc_coord=aux_enc_coord, aux_enc_score=aux_enc_score, 
                            batch_pseudo_instances=batch_pseudo_instances)

        head_inputs_dict = self.forward_transformer(visual_features, text_dict, batch_data_samples, aux_dict)    
        head_inputs_dict['text_token_mask_chunked'] = new_text_token_mask_chunked    
        head_inputs_dict['token_positive_maps'] = self.token_positive_maps 

        if 'dn_meta' in ori_head_inputs_dict.keys():
            ori_head_inputs_dict.pop('dn_meta')
        if 'enc_outputs_class' in ori_head_inputs_dict.keys():
            ori_head_inputs_dict.pop('enc_outputs_class')
            ori_head_inputs_dict.pop('enc_outputs_coord')

        losses = self.bbox_head.loss(**head_inputs_dict, **ori_head_inputs_dict, 
                                        batch_data_samples=batch_data_samples)
        return losses
    
