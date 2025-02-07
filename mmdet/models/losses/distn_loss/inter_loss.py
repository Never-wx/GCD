import torch
import torch.nn.functional as F
from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, bbox_overlaps, bbox2roi
from mmdet.models.utils import multi_apply
from mmengine.structures import InstanceData

def pdist(e, dist_mode="l2", eps=1e-6):
    dist_mode = dist_mode.lower()
    assert dist_mode in ["l1", "l2"], dist_mode
    N = e.shape[0]
    if dist_mode == "l1":
        res = torch.abs(e.unsqueeze(1) - e.unsqueeze(0)).clamp(min=eps)
    elif dist_mode == 'l2':
        e_square = e.pow(2).sum(dim=1)
        prod = torch.matmul(e, e.T)
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
    else:
        raise NotImplementedError  
    
    res = res.clone()
    res[range(N), range(N)] = 0
    mean_res = (res[res>0].mean()).clamp(min=eps)
    res_norm = res / mean_res
    return res_norm

def inter_text_relation(ori_token_positive_maps, text_feats, ori_text_feats):
    # text_prototype
    text_prototype_percls_list = []
    ori_text_prototype_percls_list = []
    for k, pos in ori_token_positive_maps.items():
        batch_text_i = text_feats[:, pos]
        batch_ori_text_i = ori_text_feats[:, pos]
        batch_text_i = torch.mean(batch_text_i, dim=1)    # [B,256]
        batch_ori_text_i = torch.mean(batch_ori_text_i, dim=1)    # [B,256]
        text_prototype = torch.mean(batch_text_i, dim=0)    # [1, 256]
        ori_text_prototype = torch.mean(batch_ori_text_i, dim=0)    
        text_prototype_percls_list.append(text_prototype)     # [num_cls, 256]
        ori_text_prototype_percls_list.append(ori_text_prototype)
    text_prototype_percls = torch.stack(text_prototype_percls_list, dim=0)
    ori_text_prototype_percls = torch.stack(ori_text_prototype_percls_list, dim=0)     

    ### interclass global text relation loss
    norm_text_diff_matrix = pdist(text_prototype_percls, dist_mode='l2')
    ori_norm_text_diff_matrix = pdist(ori_text_prototype_percls, dist_mode='l2')  
    # text_relation_loss = self.loss_textfeat_kd(norm_text_diff_matrix, ori_norm_text_diff_matrix)  
    return text_prototype_percls, ori_text_prototype_percls, norm_text_diff_matrix, ori_norm_text_diff_matrix

def inter_text_relation_partial(token_positive_maps, ori_token_positive_maps,
                                unique_pseudo_labels, ori_pseudo_labels_list, 
                                text_feats, ori_text_feats,
                                batch_cls_scores, batch_ori_cls_scores,
                                batch_query_feats, weighted=True):
    
    B,N,D = batch_query_feats.size()       
    if unique_pseudo_labels.size(0)>0:            
        total_cls_scores = batch_cls_scores.reshape(-1, D)    # [B*N, D]
        ori_total_cls_scores = batch_ori_cls_scores.reshape(-1, D)      
        total_output_score = convert_grounding_to_cls_scores(logits=total_cls_scores.sigmoid()[None],
                                                            positive_maps=[token_positive_maps])[0]       
        ori_total_output_score = convert_grounding_to_cls_scores(logits=ori_total_cls_scores.sigmoid()[None],
                                                                positive_maps=[ori_token_positive_maps])[0]                 
        total_weights, total_labels = torch.max(total_output_score, dim=1)
        ori_total_weights, ori_total_labels = torch.max(ori_total_output_score, dim=1)
        
        text_prototype_percls_list = []
        ori_text_prototype_percls_list = []
        for cls in unique_pseudo_labels:
            idx_cls = (total_labels==cls)
            ori_idx_cls = (ori_total_labels==cls)
            if torch.any(idx_cls):
                text_pos = ori_token_positive_maps[cls.item()+1]
                text_cls = []
                ori_text_cls = []
                # only collect text in current img
                for i, cur_cls in enumerate(ori_pseudo_labels_list):
                    if cls in cur_cls:
                        mean_text = torch.mean(text_feats[i, text_pos], dim=0)  # [256]
                        mean_ori_text = torch.mean(ori_text_feats[i, text_pos], dim=0)  # [256]
                        text_cls.append(mean_text)
                        ori_text_cls.append(mean_ori_text)
                text_cls = torch.stack(text_cls, dim=0)  # [n_cls, 256]
                ori_text_cls = torch.stack(ori_text_cls, dim=0)                   
                text_prototype = torch.mean(text_cls, dim=0)    # [256]
                ori_text_prototype = torch.mean(ori_text_cls, dim=0)    
                text_prototype_percls_list.append(text_prototype)     # num_cls * [256]
                ori_text_prototype_percls_list.append(ori_text_prototype)  
        
    ### interclass global text relation loss
    if unique_pseudo_labels.size(0)>2 and len(text_prototype_percls_list) > 2:
        text_prototype_percls = torch.stack(text_prototype_percls_list, dim=0)
        ori_text_prototype_percls = torch.stack(ori_text_prototype_percls_list, dim=0)    
        norm_text_diff_matrix = pdist(text_prototype_percls, dist_mode='l2')
        ori_norm_text_diff_matrix = pdist(ori_text_prototype_percls, dist_mode='l2')  
        # text_relation_loss = self.loss_textfeat_kd(norm_text_diff_matrix, ori_norm_text_diff_matrix)  
    else:
        text_prototype_percls = []
        ori_text_prototype_percls = []
        norm_text_diff_matrix = []
        ori_norm_text_diff_matrix = []

    return text_prototype_percls, ori_text_prototype_percls, norm_text_diff_matrix, ori_norm_text_diff_matrix

def inter_query_relation(token_positive_maps, ori_token_positive_maps, unique_pseudo_labels, 
                         batch_cls_scores, batch_ori_cls_scores,
                         batch_query_feats, batch_ori_query_feats, weighted=True):
    
    assert batch_query_feats.size() == batch_ori_query_feats.size()
    B, N, D = batch_query_feats.size()  
    if unique_pseudo_labels.size(0) > 0:            
        total_query_feats =  batch_query_feats.reshape(-1, D) # [B*N, 256]
        total_cls_scores = batch_cls_scores.reshape(-1, D)    # [B*N, D]
        ori_total_query_feats = batch_ori_query_feats.reshape(-1, D)
        ori_total_cls_scores = batch_ori_cls_scores.reshape(-1, D)      

        total_output_score = convert_grounding_to_cls_scores(logits=total_cls_scores.sigmoid()[None],
                                                            positive_maps=[token_positive_maps])[0]       
        ori_total_output_score = convert_grounding_to_cls_scores(logits=ori_total_cls_scores.sigmoid()[None],
                                                                positive_maps=[ori_token_positive_maps])[0]                 
        total_weights, total_labels = torch.max(total_output_score, dim=1)
        ori_total_weights, ori_total_labels = torch.max(ori_total_output_score, dim=1)
        
        # query_feats_percls_list = []
        query_prototype_percls_list = []
        # ori_query_feats_percls_list = []
        ori_query_prototype_percls_list = []
        for cls in unique_pseudo_labels:
            idx_cls = (total_labels==cls)
            ori_idx_cls = (ori_total_labels==cls)
            if torch.any(idx_cls):
                query_feats_cls = total_query_feats[idx_cls]
                ori_query_feats_cls = ori_total_query_feats[ori_idx_cls]
                if weighted:
                    weights = (total_weights[idx_cls] / total_weights[idx_cls].sum()).unsqueeze(1)
                    ori_weights = (ori_total_weights[ori_idx_cls] / ori_total_weights[ori_idx_cls].sum()).unsqueeze(1)
                    query_prototype_cls = (weights * query_feats_cls).sum(dim=0)
                    ori_query_prototype_cls = (ori_weights * ori_query_feats_cls).sum(dim=0)
                else:
                    query_prototype_cls = torch.mean(query_feats_cls, dim=0)
                    ori_query_prototype_cls = torch.mean(ori_query_feats_cls, dim=0)
                query_prototype_percls_list.append(query_prototype_cls) 
                ori_query_prototype_percls_list.append(ori_query_prototype_cls)

    ## interclass query relation loss, relation matrix = I where class <=2 
    if unique_pseudo_labels.size(0) > 2 and len(query_prototype_percls_list) > 2:
        query_prototype_percls = torch.stack(query_prototype_percls_list, dim=0)
        ori_query_prototype_percls = torch.stack(ori_query_prototype_percls_list, dim=0)
        norm_query_diff_matrix = pdist(query_prototype_percls)
        ori_norm_query_diff_matrix = pdist(ori_query_prototype_percls)  
        # query_relation_loss = self.loss_imgfeat_kd(norm_query_diff_matrix, ori_norm_query_diff_matrix) 
    else:
        norm_query_diff_matrix = []
        ori_norm_query_diff_matrix = []
    return norm_query_diff_matrix, ori_norm_query_diff_matrix

