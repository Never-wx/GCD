# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmengine.model import constant_init
from mmengine.structures import InstanceData
from torch import Tensor
from mmengine import Config
from mmdet.models.losses import QualityFocalLoss
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, bbox_overlaps
from mmdet.utils import InstanceList, ConfigType, reduce_mean,  OptInstanceList
from ..utils import multi_apply
from ..layers import inverse_sigmoid
from mmdet.structures.bbox import bbox2roi
from .atss_vlfusion_head import convert_grounding_to_cls_scores
# from .gdino_head_inc_distn import GroundingDINOHead_inc_distn
from .gdino_head_inc import GroundingDINOHead_inc
from mmdet.models.losses.distn_loss import inter_query_relation, inter_text_relation_partial

@MODELS.register_module()
class GroundingDINOHead_inc_gcd(GroundingDINOHead_inc):
    """Head of the Grounding DINO: Marrying DINO with Grounded Pre-Training for
    Open-Set Object Detection.

    Args:
        contrastive_cfg (dict, optional): Contrastive config that contains
          keys like ``max_text_len``. Defaults to dict(max_text_len=256).
    """

    def __init__(self, distn_cfg, **kwargs):
        super().__init__(**kwargs)
        self.distn_cfg = distn_cfg
        if 'label_distn' not in self.distn_cfg:
            self.distn_cfg.label_distn = Config._dict_to_config_dict_lazy(dict(type='None'))
        if 'feat_distn' not in self.distn_cfg:
            self.distn_cfg.feat_distn = Config._dict_to_config_dict_lazy(dict(type='None'))
        if 'query_distn' not in self.distn_cfg:
            self.distn_cfg.query_distn = Config._dict_to_config_dict_lazy(dict(type='None'))

        if self.distn_cfg.label_distn.type == 'topk_pseudo' or self.distn_cfg.label_distn.type == 'threshold_pseudo':             
            self.sigma = self.distn_cfg.label_distn.sigma
            self.label_iou_th = self.distn_cfg.label_distn.label_iou_th

        if self.distn_cfg.feat_distn.type != 'None':
            if 'img_loss' in self.distn_cfg.feat_distn:
                self.loss_imgfeat_kd = MODELS.build(self.distn_cfg.feat_distn.img_loss)
            if 'text_loss' in self.distn_cfg.feat_distn:
                self.loss_textfeat_kd = MODELS.build(self.distn_cfg.feat_distn.text_loss)

        if 'loss_ld' in self.distn_cfg.label_distn:
            self.loss_ld = MODELS.build(self.distn_cfg.label_distn.loss_ld)        
        else:
            self.distn_cfg.label_distn.loss_ld = dict()
            self.distn_cfg.label_distn.loss_ld.type = 'None'

    @torch.no_grad()
    def generate_pseudo_label(self, all_layers_cls_scores, all_layers_bbox_preds, ori_text_token_mask,
                               new_text_token_mask, batch_data_samples, ori_token_positive_maps, test=None, mode='None'):
        batch_img_metas = []
        batch_gt_instances = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        # select last layer's topk response for distillation
        last_layer_ori_cls_scores = all_layers_cls_scores[-1]
        last_layer_ori_bbox_preds = all_layers_bbox_preds[-1]

        B,L,_ = last_layer_ori_cls_scores.size()

        ori_text_masks = ori_text_token_mask.new_zeros(
            (ori_text_token_mask.size(0), self.max_text_len))
        
        # get new/old model's cls response accroding to old text mask
        ori_text_masks[:, :ori_text_token_mask.size(1)] = ori_text_token_mask
        ori_text_mask = (ori_text_masks > 0).unsqueeze(1)
        ori_text_mask = ori_text_mask.repeat(1, last_layer_ori_cls_scores.size(1), 1)

        valid_ori_cls_scores = torch.masked_select(last_layer_ori_cls_scores, ori_text_mask).contiguous().view(B, L, -1)

        if len(ori_text_token_mask.size()) > 1:
            ori_text_token_mask = ori_text_token_mask[0]
        if len(new_text_token_mask.size()) > 1:
            new_text_token_mask = new_text_token_mask[0]

        topk_query, batch_pseudo_instances, batch_prob_maxval, batch_prob_maxidx = \
            multi_apply(self.generate_pesudo_label_single,
                        last_layer_ori_cls_scores,
                        valid_ori_cls_scores,
                        last_layer_ori_bbox_preds,
                        batch_gt_instances,
                        batch_img_metas,
                        ori_text_token_mask=ori_text_token_mask,
                        new_text_token_mask=new_text_token_mask,
                        ori_token_positive_maps=ori_token_positive_maps,
                        test=test)
        
        batch_all_instances = []
        for gt_instances, pseudo_instances in zip(batch_gt_instances, batch_pseudo_instances):                        
            all_bboxes = torch.cat((gt_instances.bboxes, pseudo_instances.bboxes), 0)
            all_positive_maps = torch.cat((gt_instances.positive_maps, pseudo_instances.positive_maps), 0)
            all_labels = torch.cat((gt_instances.labels, pseudo_instances.labels), 0)
            all_text_token_masks = new_text_token_mask.repeat(all_labels.size(0), 1)   # [num, all_text_len]             
            all_instances = InstanceData(labels=all_labels, positive_maps=all_positive_maps, 
                                        text_token_mask=all_text_token_masks, bboxes=all_bboxes)
            batch_all_instances.append(all_instances)

        return topk_query, batch_pseudo_instances, batch_all_instances

    @torch.no_grad()
    def generate_pesudo_label_single(self, ori_cls_scores: Tensor, valid_ori_cls_scores: Tensor, 
                                    ori_bbox_preds: Tensor, gt_instances: InstanceData, img_metas: dict,
                                    ori_text_token_mask: List[Tensor], new_text_token_mask: List[Tensor],
                                    ori_token_positive_maps: Tensor, test=False):
        if test:    ### just for debug
            gt_text_token_mask = ori_text_token_mask.repeat(gt_instances.labels.size(0), 1)
            gt_positive_maps = torch.zeros([1, 256]).repeat(gt_instances.labels.size(0), 1).cuda()
            gt_instances.positive_maps = gt_positive_maps
            gt_bboxes, gt_labels = gt_instances.bboxes, gt_instances.labels
        else:
            gt_bboxes, gt_positive_maps, gt_text_token_mask, gt_labels = \
                gt_instances.bboxes, gt_instances.positive_maps, gt_instances.text_token_mask, gt_instances.labels 

        img_h, img_w = img_metas['img_shape']
        factor = ori_bbox_preds.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        num_bboxes = ori_bbox_preds.size(0)  # [900, 4]

        if gt_text_token_mask.size(0) != 0 and gt_labels.max() >= self.trunc_class[0]:    # not empty sample        
            # convert positive_map to label
            ori_output_score = convert_grounding_to_cls_scores(logits=valid_ori_cls_scores.sigmoid()[None],
                positive_maps=[ori_token_positive_maps])[0]        # [900, num_cls]
            
            # select pred with high confidence
            prob_maxval, prob_maxidx = ori_output_score.max(dim=-1)
            # prob_maxval, _ = valid_ori_cls_scores.sigmoid().max(dim=-1)
            if self.distn_cfg.label_distn.type == 'topk_pseudo':  # topk select
                topk_value, topk_idx = torch.topk(prob_maxval, self.sigma, dim=-1)
            elif self.distn_cfg.label_distn.type == 'threshold_pseudo':     # threshold select
                topk_idx = torch.where(prob_maxval >= self.sigma)[0]                                              
            else:
                raise ValueError('not implement')
            # generate pseudo_label
            ref_labels = prob_maxidx[topk_idx]

            # generate positive_map 
            # ref_positive_map = torch.zeros_like(ori_cls_scores[topk_idx])   # [topk, 256]
            # for i in range(ref_labels.size(0)):
            #     positive_idx = ori_token_positive_maps[ref_labels[i].item()+1]
            #     ref_positive_map[i, positive_idx] = 1   # hard pseudo label
            
            if self.distn_cfg.label_distn.mode == 'response':
                ref_positive_map = torch.zeros_like(ori_cls_scores[topk_idx])   # [topk, 256]
                valid_len = valid_ori_cls_scores.size(1) - 1    # last pos mean EOS
                ref_positive_map[:, :valid_len] = valid_ori_cls_scores[topk_idx][:, :valid_len]
            else:   # hardlabel
                ref_positive_map = torch.zeros_like(ori_cls_scores[topk_idx])   # [topk, 256]
                for i in range(ref_labels.size(0)):
                    positive_idx = ori_token_positive_maps[ref_labels[i].item()+1]
                    ref_positive_map[i, positive_idx] = 1   # hard pseudo label 

            # convert bbox_pred from xywh, normalized to xyxy, unnormalized
            ref_box_list = ori_bbox_preds[topk_idx] # [topk, 4]
            ref_box_list = bbox_cxcywh_to_xyxy(ref_box_list)
            ref_box_list = ref_box_list * factor
            ref_box_list[:, 0::2].clamp_(min=0, max=img_w)
            ref_box_list[:, 1::2].clamp_(min=0, max=img_h)

            # return boxes in original image
            iou_list = bbox_overlaps(ref_box_list, gt_bboxes)
            ioumax_val, ioumax_idx = torch.max(iou_list, dim=1)  

            # avoid overlap with gt
            gt_include_list = torch.where(ioumax_val>self.label_iou_th, False, True)
            pseudo_bboxes = ref_box_list[gt_include_list]
            pseudo_positive_maps = ref_positive_map[gt_include_list]
            pseudo_labels = ref_labels[gt_include_list]
            # avoid duplicate bbox
            
            # pseudo_text_token_masks = new_text_token_mask.repeat(pseudo_labels.size(0), 1)  # [num, ori_text_len]
            if self.distn_cfg.label_distn.mode == 'response':
                pseudo_text_token_masks = ori_text_token_mask[:valid_len].repeat(pseudo_labels.size(0), 1)  
            else:
                pseudo_text_token_masks = new_text_token_mask.repeat(pseudo_labels.size(0), 1)  # [num, ori_text_len]

            if pseudo_bboxes.dtype == torch.float16:
                pseudo_bboxes = pseudo_bboxes.to(torch.float32)

            topk_idx = topk_idx[gt_include_list]
                
            gt_instances = InstanceData(labels=pseudo_labels, positive_maps=pseudo_positive_maps, 
                                        text_token_mask=pseudo_text_token_masks, bboxes=pseudo_bboxes)
        else:
            topk_idx = None
            prob_maxval = ori_cls_scores.new_tensor([])
            prob_maxidx = ori_cls_scores.new_tensor([])
        
        return topk_idx, gt_instances, prob_maxval, prob_maxidx

    def loss(self, new_head_inputs_dict, 
             old_head_inputs_dict, 
             ori_head_inputs_dict, 
             batch_data_samples) -> dict:       
         
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        # self.ori_text_mask = ori_head_inputs_dict['ori_text_token_mask']  # text_mask for old class 
        self.ori_topk_query = ori_head_inputs_dict['ori_topk_query']
        self.ori_token_positive_maps = ori_head_inputs_dict['ori_token_positive_maps'] 

        self.token_positive_maps = new_head_inputs_dict['token_positive_maps']
        self.text_masks = new_head_inputs_dict['text_token_mask']   # text mask for new class  
        self.ori_text_masks = old_head_inputs_dict['text_token_mask']   # text_mask for old class 

        new_outs = self(new_head_inputs_dict['hidden_states'], 
                        new_head_inputs_dict['references'], 
                        new_head_inputs_dict['memory_text'], 
                        new_head_inputs_dict['text_token_mask'])  
        
        old_outs = self(old_head_inputs_dict['hidden_states'], 
                        old_head_inputs_dict['references'], 
                        old_head_inputs_dict['memory_text'], 
                        old_head_inputs_dict['text_token_mask'])  
        
        new_enc_cls_scores = new_head_inputs_dict['enc_outputs_class']
        new_enc_bbox_preds = new_head_inputs_dict['enc_outputs_coord']
        dn_meta = new_head_inputs_dict['dn_meta']

        # old_enc_cls_scores = old_head_inputs_dict['enc_outputs_class']
        # old_enc_bbox_preds = new_head_inputs_dict['enc_outputs_coord']
        hidden_states = old_head_inputs_dict['hidden_states']
        memory_text = old_head_inputs_dict['memory_text']

        all_layers_ori_cls_scores = ori_head_inputs_dict['all_layers_ori_cls_scores']
        all_layers_ori_bbox_preds = ori_head_inputs_dict['all_layers_ori_bbox_preds']
        ori_hidden_states = ori_head_inputs_dict['ori_hidden_states']
        ori_memory_text = ori_head_inputs_dict['ori_memory_text']
        # ori_enc_outputs_class = ori_head_inputs_dict['enc_outputs_class']
        # ori_enc_outputs_coord = ori_head_inputs_dict['enc_outputs_coord']
        if 'batch_pseudo_instances' in ori_head_inputs_dict.keys():
            batch_pseudo_instances = ori_head_inputs_dict['batch_pseudo_instances']
            batch_all_instances = ori_head_inputs_dict['batch_all_instances']
        else:
            batch_pseudo_instances = None
            batch_all_instances = None
        # if 'enc_feat_dict' in old_head_inputs_dict.keys():
        #     enc_feat_dict = old_head_inputs_dict['enc_feat_dict']
        #     ori_enc_feat_dict = ori_head_inputs_dict['enc_feat_dict']
        # else:
        #     enc_feat_dict = None
        #     ori_enc_feat_dict = None

        detr_loss_inputs = new_outs + (new_enc_cls_scores, new_enc_bbox_preds, 
                                       dn_meta, batch_gt_instances, batch_img_metas, batch_all_instances)
        
        distn_loss_inputs = old_outs + (batch_gt_instances, batch_img_metas, hidden_states, memory_text) + \
                                        (all_layers_ori_cls_scores, all_layers_ori_bbox_preds,
                                         batch_pseudo_instances, batch_all_instances, ori_hidden_states,
                                         ori_memory_text)

        loss_dict_old = self.loss_by_feat_old(*distn_loss_inputs)
        loss_dict_new = self.loss_by_feat_new(*detr_loss_inputs)
        loss_dict_new.update(loss_dict_old)
        return loss_dict_new    

    def loss_by_feat_old(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        hidden_states: Tensor = None,
        memory_text: Tensor = None,
        all_layers_ori_cls_scores: Tensor = None, # original model input for distn
        all_layers_ori_bbox_preds: Tensor = None, # original model input for distn
        batch_pseudo_instances: OptInstanceList = None,
        batch_all_instances: OptInstanceList = None,
        ori_hidden_states: Tensor = None,
        ori_memory_text: Tensor = None,
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> Dict[str, Tensor]:

        loss_dict = dict()
        if batch_pseudo_instances is None:
            batch_all_instances = batch_gt_instances
        
        # ===== query distn loss ===== 
        if self.distn_cfg.query_distn.type == 'seperate_queryinit':
            # num_matching_query = self.distn_cfg.query_distn.num_matching_query
            num_aux_queries = self.distn_cfg.query_distn.num_aux_query            
            all_layers_aux_cls_scores = all_layers_cls_scores
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_aux_bbox_preds = all_layers_bbox_preds
            all_layers_matching_bbox_preds = all_layers_bbox_preds
            all_layers_ori_cls_scores = all_layers_ori_cls_scores[:, :, :num_aux_queries]
            all_layers_ori_bbox_preds = all_layers_ori_bbox_preds[:, :, :num_aux_queries]

        # ===== logits distn loss =====    
        ld_losses_cls, ld_losses_bbox, ld_losses_iou = multi_apply(
            self.loss_by_feat_ld_distn_single,
            all_layers_aux_cls_scores,
            all_layers_aux_bbox_preds,
            all_layers_ori_cls_scores,
            all_layers_ori_bbox_preds,
            # batch_gt_instances=batch_all_instances,
            batch_gt_instances=batch_gt_instances,
            batch_pseudo_instances=batch_pseudo_instances,
            batch_img_metas=batch_img_metas) 
            
        # loss from the last decoder layer        
        loss_dict['loss_ld_cls'] = ld_losses_cls[-1]
        loss_dict['loss_ld_bbox'] = ld_losses_bbox[-1]
        loss_dict['loss_ld_iou'] = ld_losses_iou[-1]      
        # detr loss and distn loss for every decoder layer
        num_dec_layer = 0
        for ld_loss_cls_i, ld_loss_bbox_i, ld_loss_iou_i in \
            zip(ld_losses_cls[:-1], ld_losses_bbox[:-1], ld_losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.ld_loss_cls'] = ld_loss_cls_i
            loss_dict[f'd{num_dec_layer}.ld_loss_bbox'] = ld_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.ld_loss_iou'] = ld_loss_iou_i
            num_dec_layer += 1   

            # enc_ld_loss_cls, enc_ld_loss_bbox, enc_ld_loss_iou = \
            #     self.loss_by_feat_ld_distn_single(old_enc_cls_scores, old_enc_bbox_preds,                                                                                       
            #                                       ori_enc_outputs_class, ori_enc_outputs_coord,
            #                                       batch_pseudo_instances=batch_pseudo_instances,
            #                                       batch_img_metas=batch_img_metas) 
            
            # loss_dict['enc_ld_loss_cls'] = enc_ld_loss_cls
            # loss_dict['enc_ld_loss_bbox'] = enc_ld_loss_bbox
            # loss_dict['enc_ld_loss_iou'] = enc_ld_loss_iou
                                                            
        # ===== feat distn loss ===== 
        if self.distn_cfg.feat_distn.type == 'inter-class':       
            inter_text_loss = None
            inter_query_loss = None
            
            inter_text_loss, inter_query_loss = \
                self.inter_feat_distn_single(hidden_states[-1], ori_hidden_states[-1],
                                            all_layers_matching_cls_scores[-1], all_layers_matching_bbox_preds[-1], 
                                            all_layers_ori_cls_scores[-1], all_layers_ori_bbox_preds[-1], 
                                            memory_text, ori_memory_text, 
                                            batch_pseudo_instances, batch_all_instances, batch_img_metas)
            
            if inter_text_loss is not None:
                loss_dict['inter_text_loss'] = inter_text_loss
            if inter_query_loss is not None:
                loss_dict['inter_query_loss'] = inter_query_loss 

        return loss_dict

    def loss_by_feat_new(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        new_enc_cls_scores: Tensor,
        new_enc_bbox_preds: Tensor,
        dn_meta: Dict[str, int],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_all_instances: InstanceList,
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> Dict[str, Tensor]:
       
        loss_dict = dict()
        # extract denoising and matching part of outputs
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds) = \
            self.split_outputs(all_layers_cls_scores, all_layers_bbox_preds, dn_meta)   

        # if self.distn_cfg.label_distn.mode == 'seperate_logits':
        #     for gt_instances in batch_gt_instances:
        #         num_label = len(gt_instances.labels) 
        #         gt_instances.text_token_mask = self.text_masks[0].repeat(num_label, 1)

        # ===== detr loss ===== 
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_by_feat_single,
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            batch_gt_instances=batch_all_instances,
            batch_img_metas=batch_img_metas)

        # loss from the last decoder layer        
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]      
        # detr loss and distn loss for every decoder layer
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in \
            zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1   
        
        # loss of proposal generated from encode feature map.               
        if new_enc_cls_scores is not None:
            #TODO: encoder output distillation
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    new_enc_cls_scores, new_enc_bbox_preds,
                    batch_gt_instances=batch_all_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds,
                batch_gt_instances=batch_gt_instances, batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
                                  dn_losses_iou[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
        return loss_dict

    def loss_by_feat_ld_distn_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                                    ori_cls_scores: Tensor, ori_bbox_preds: Tensor,
                                    batch_pseudo_instances: InstanceList,
                                    batch_gt_instances: InstanceList,
                                    batch_img_metas: List[dict],
                                    weighted=True) -> Tuple[Tensor]:
        
        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # filter ori_bboxes overlaped with gt
        overlap_list = []
        for instance, ori_bboxes, factor, img_metas in zip(batch_gt_instances, ori_bbox_preds, factors, 
                                                           batch_img_metas):
            if len(instance.labels) > 0:
                img_h, img_w, = img_meta['img_shape']
                gt_bboxes = instance.bboxes
                ori_bboxes = bbox_cxcywh_to_xyxy(ori_bboxes)
                ori_bboxes = ori_bboxes * factor
                ori_bboxes[:, 0::2].clamp_(min=0, max=img_w)
                ori_bboxes[:, 1::2].clamp_(min=0, max=img_h)
                iou_list = bbox_overlaps(ori_bboxes, gt_bboxes)
                ioumax_val, ioumax_idx = torch.max(iou_list, dim=1) 
                invalid_bbox = torch.where(ioumax_val>self.label_iou_th, True, False)
                overlap_list.append(invalid_bbox)
            else:
                overlap_list.append(ori_bboxes.new_ones(ori_bboxes.size(0)).bool())

        overlap_list = torch.stack(overlap_list, dim=0)    
        
        max_scores, _ = ori_cls_scores.sigmoid().max(dim=-1)    # [B,900]   
        
        if weighted:       
            label_weights = torch.zeros_like(max_scores)
            bbox_weights = torch.zeros_like(max_scores)    
            valid_mask_list = torch.zeros_like(max_scores)    
            for i, score in enumerate(max_scores):
                cls_thr = score.mean() + 2 * score.std()
                valid_mask = score > cls_thr
                #label_weights[i][valid_mask] = score[valid_mask]
                #bbox_weights[i][valid_mask] = score[valid_mask]
                label_weights[i][valid_mask] = 1.0
                bbox_weights[i][valid_mask] = 1.0
                valid_mask_list[i] = valid_mask
            label_weights[overlap_list] = 0.0  
            bbox_weights[overlap_list] = 0.0
            valid_mask_list[overlap_list] = 0.0
            # for i, bbox_weight in enumerate(bbox_weights):
            #     pos_inds = self.ori_topk_query[i]
            #     if pos_inds is not None:
            #         pos_inds = pos_inds[pos_inds<bbox_weight.size(0)]
            #         bbox_weight[pos_inds] = 1.0
        else:
            label_weights = torch.ones_like(max_scores)
            bbox_weights = torch.ones_like(max_scores)    
            valid_mask_list = torch.ones_like(max_scores)    
            label_weights[overlap_list] = 0.0  
            bbox_weights[overlap_list] = 0.0
            valid_mask_list[overlap_list] = 0.0   
        
        ori_text_masks = self.ori_text_masks
        assert (ori_text_masks.dim() == 2)
        text_masks = ori_text_masks.new_zeros(
            (ori_text_masks.size(0), self.max_text_len))
        text_masks[:, :ori_text_masks.size(1)] = ori_text_masks
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, cls_scores.size(1), 1)
        cls_scores = torch.masked_select(cls_scores, text_mask).contiguous()
        ori_cls_scores = torch.masked_select(ori_cls_scores, text_mask).contiguous()
        # label_weights = torch.masked_select(label_weights, text_mask).contiguous()

        # num_total_pos = 0
        # for instance in batch_pseudo_instances:
        #     num_total_pos += len(instance.labels)
        num_total_pos = valid_mask_list.sum()
        cls_avg_factor = num_total_pos * 1.0
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)


        # valid_ori_text_len = self.ori_text_masks[0].sum()

        if self.distn_cfg.label_distn.loss_ld.type == 'L2Loss':
            label_weights = label_weights[..., None].repeat(1, 1, text_mask.size(-1))
            label_weights = torch.masked_select(label_weights, text_mask)   # 900 * 195, all 1   
            loss_cls = self.loss_ld(cls_scores, ori_cls_scores, label_weights, avg_factor=cls_avg_factor)
            
        elif self.distn_cfg.label_distn.loss_ld.type == 'KnowledgeDistillationKLDivLoss':
            valid_ori_text_len = self.ori_text_masks[0].sum()
            cls_scores = cls_scores.view(-1, valid_ori_text_len)    # [batch * num_query, valid_ori_text_len]
            labels = ori_cls_scores.view(-1, valid_ori_text_len) 
            loss_cls = self.loss_ld(cls_scores, labels, label_weights.flatten(), avg_factor=cls_avg_factor)         
 
        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss

        if ori_bbox_preds.dtype == torch.float16:
            ori_bbox_preds = ori_bbox_preds.to(torch.float32)

        bbox_preds = bbox_preds.reshape(-1, 4)
        ori_bbox_preds = ori_bbox_preds.reshape(-1, 4)
        bbox_weights = bbox_weights.unsqueeze(-1).repeat(1, 1, bbox_preds.size(-1))
        bbox_weights = bbox_weights.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(ori_bbox_preds) * factors
        # bboxes[:, 0::2].clamp_(min=0, max=img_w)
        # bboxes[:, 1::2].clamp_(min=0, max=img_h)
        # bboxes_gt[:, 0::2].clamp_(min=0, max=img_w)
        # bboxes_gt[:, 1::2].clamp_(min=0, max=img_h)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=cls_avg_factor)
        
        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, ori_bbox_preds, bbox_weights, avg_factor=cls_avg_factor)

        return loss_cls, loss_bbox, loss_iou 
           
    def loss_by_feat_single_new(self, cls_scores: Tensor, bbox_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]      # [1, 900, 256]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]      # [1, 900, 4]

        # select pos bbox for backward update
        with torch.no_grad():
            cls_reg_targets = self.get_targets(cls_scores_list,
                                               bbox_preds_list,
                                               batch_gt_instances,
                                               batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # # select query belonging to new text
        # batch_max_cls_scores, batch_max_inds = cls_scores.sigmoid().max(dim=-1)
        # new_text_inds = batch_max_inds >= len(self.ori_text_mask[0]) - 1
        # label_weights = new_text_inds.float()

        # ===== this change =====
        # Loss is not computed for the padded regions of the text.
        assert (self.text_masks.dim() == 2)
        text_masks = self.text_masks.new_zeros(
            (self.text_masks.size(0), self.max_text_len))
        text_masks[:, :self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, cls_scores.size(1), 1)
        cls_scores = torch.masked_select(cls_scores, text_mask).contiguous()

        # (num_query * valid_text_len = 900 * 195 = 175500, neg query all zero, pos query one in corresponding text pos)
        labels = torch.masked_select(labels, text_mask) 
        label_weights = label_weights[..., None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)   # 900 * 195, all 1

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight  # self.bg_cls_weight = 0
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            raise NotImplementedError(
                'QualityFocalLoss for GroundingDINOHead is not supported yet.')
        else:
            loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou
    
    def inter_feat_distn_single(self, batch_query_feats, batch_ori_query_feats,
                                    batch_cls_scores, batch_bbox_preds, 
                                    batch_ori_cls_scores, batch_ori_bbox_preds, 
                                    batch_text_feats, batch_ori_text_feats,
                                    batch_pseudo_instances, batch_all_instances, batch_img_metas):
        ori_pseudo_labels_list = []
        all_labels_list = []

        for pseudo_instance in batch_pseudo_instances:
            ori_pseudo_labels_list.append(pseudo_instance.labels)
        unique_pseudo_labels = torch.unique(torch.cat(ori_pseudo_labels_list, dim=0))

        for all_instances in batch_all_instances:
            all_labels_list.append(all_instances.labels)
        unique_labels = torch.unique(torch.cat(all_labels_list, dim=0))

        norm_text_diff_matrix = []
        norm_query_diff_matrix = []
        ori_norm_query_diff_matrix = []
        intra_distance = []

        if self.distn_cfg.feat_distn.subtype == 'opt1':
            # # text_relation
            # _, _, norm_text_diff_matrix, ori_norm_text_diff_matrix = \
            #     inter_text_relation(self.ori_token_positive_maps, batch_text_feats, batch_ori_text_feats)
            # query_relation
            norm_query_diff_matrix, ori_norm_query_diff_matrix = \
                inter_query_relation(self.ori_token_positive_maps, self.ori_token_positive_maps, unique_pseudo_labels, 
                                    batch_cls_scores, batch_ori_cls_scores,
                                    batch_query_feats, batch_ori_query_feats)
            # text_relation
            _, _, norm_text_diff_matrix, ori_norm_text_diff_matrix = \
                inter_text_relation_partial(self.ori_token_positive_maps, self.ori_token_positive_maps,
                            unique_pseudo_labels, ori_pseudo_labels_list, 
                            batch_text_feats, batch_ori_text_feats,
                            batch_cls_scores, batch_ori_cls_scores,
                            batch_query_feats)            
                    
        text_relation_loss = None
        query_relation_loss = None

        if len(ori_norm_query_diff_matrix) > 0: # inter_query
            query_relation_loss = self.loss_imgfeat_kd(norm_query_diff_matrix, ori_norm_query_diff_matrix)   
        
        if len(norm_text_diff_matrix) > 0:                
            text_relation_loss = self.loss_textfeat_kd(norm_text_diff_matrix, ori_norm_text_diff_matrix)  
    
        return text_relation_loss, query_relation_loss

