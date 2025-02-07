# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Union
import copy
import torch.nn as nn
import torch
from collections import OrderedDict
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmdet.registry import HOOKS
from mmengine.runner.checkpoint import _load_checkpoint
DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class Increment_distn_hook(Hook):
    def __init__(self):
        pass

    def after_load_checkpoint(self, runner: Runner, checkpoint):
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        self.train_weights_transform(model, checkpoint)
        
    def before_save_checkpoint(self, runner, checkpoint):
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
            pop_key_list = []
            for k, v in checkpoint.items():
                if 'ori_model' in k:
                    pop_key_list.append(k)                    
            for k in pop_key_list:
                checkpoint.pop(k)        
                
    def train_weights_transform(self, model, checkpoint):
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']        
        ori_model_weights = dict()
        
        for k, v in state_dict.items():
            new_k = 'ori_model.' + k
            ori_model_weights[new_k] = v
        state_dict.update(ori_model_weights)             

        if model.dn_cfg is not None:    
            if state_dict['dn_query_generator.label_embedding.weight'].size() != \
                model.dn_query_generator.label_embedding.weight.size():  
                added_dn_labelemb_weight = model.dn_query_generator.label_embedding.weight[model.start:, ...]        
                state_dict['dn_query_generator.label_embedding.weight'] = torch.cat(
                    (state_dict['dn_query_generator.label_embedding.weight'], added_dn_labelemb_weight.cpu()), dim=0)   


