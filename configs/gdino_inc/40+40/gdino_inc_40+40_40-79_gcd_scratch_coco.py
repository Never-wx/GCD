_base_ = '../_base_/gdino_inc_distn_coco.py'
load_from = './work_dirs/gdino_inc_40+40_0-39_scratch_coco/epoch_12.pth'

dataset_type = 'CocoIncDataset'
data_root = './data/coco/'
start = 40
end = 80
distn_cfg=dict(
    ori_config_file='configs/gdino_inc/40+40/gdino_inc_40+40_0-39_scratch_coco.py',
    future_class=False,
    label_distn=dict(
        type='threshold_pseudo', # choice = ['topk_pseudo', 'threshold_pseudo']
        mode='hardlabel', # choice = ['seperate_logits', 'hardlabel']
        loss_ld=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=1.0, T=100),
        sigma=0.4, label_iou_th=0.7,
    ),
    feat_distn=dict(
        type='inter-class', 
        subtype = 'opt1',
        img_loss=dict(type='L2Loss', loss_weight=3.0, reduction='mean'),
        text_loss=dict(type='L2Loss', loss_weight=5.0, reduction='mean'),
    ),        
    query_distn=dict(
        type='seperate_queryinit',  
        num_matching_query=900,   
        num_aux_query=900,
    ), 
)

model = dict(
    type='GroundingDINO_inc_gcd',
    num_queries=900,
    bbox_head=dict(
        type='GroundingDINOHead_inc_gcd',
        contrastive_cfg=dict(max_text_len=256, log_scale=None, bias=None),
        distn_cfg=distn_cfg,
        trunc_class=[start, end]),
    dn_cfg=None,
    distn_cfg=distn_cfg
)

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/40+40/instances_train2017_40-79.json',
        setting='full_text',     
        start=start,
        end=end))

val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        setting='full_text',
        start=start,
        end=end))

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')

test_dataloader = val_dataloader
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.00005, weight_decay=0.0001))
# learning policy
max_epochs = 12

custom_hooks = [
    dict(type='Increment_distn_hook')
]
