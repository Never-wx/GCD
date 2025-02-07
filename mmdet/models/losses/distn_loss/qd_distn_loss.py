import torch
import os
import copy
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, bbox_overlaps
from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores
from mmdet.models.layers.transformer.utils import inverse_sigmoid
from mmengine.structures import InstanceData

def generate_distn_points(distn_cfg, aux_dict):
    
    num_aux_queries = distn_cfg.query_distn.num_aux_query
    num_matching_queries = distn_cfg.query_distn.num_matching_query
    distn_points = aux_dict['aux_query']
    aux_enc_coord = aux_dict['aux_enc_coord']
    aux_enc_score = aux_dict['aux_enc_score']
    aux_reference_points = aux_dict['aux_reference']
    
    if 'noise_scale' in distn_cfg.query_distn:
        noise_scale = distn_cfg.query_distn.noise_scale
        enc_bboxes = bbox_cxcywh_to_xyxy(aux_reference_points) 
        bboxes_whwh = bbox_xyxy_to_cxcywh(enc_bboxes)[:, :, 2:].repeat(1, 1, 2)
        rand_part = torch.rand_like(enc_bboxes) * 2 - 1.0                        
        enc_bboxes += torch.mul(rand_part, bboxes_whwh) * noise_scale  # xyxy   
        enc_bboxes = enc_bboxes.clamp(min=0.0, max=1.0) ## xyxy, normalized                 
        aux_reference_points = bbox_xyxy_to_cxcywh(enc_bboxes)
    
    ## draw attn_mask between matching part and aux part
    num_total_query = num_matching_queries + num_aux_queries 
    self_attn_mask = distn_points.new_zeros([num_total_query, num_total_query]).bool()
    self_attn_mask[num_matching_queries:, :num_matching_queries] = True
    self_attn_mask[:num_matching_queries, num_matching_queries:] = True

    return distn_points, aux_reference_points, self_attn_mask
    
