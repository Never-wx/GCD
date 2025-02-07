_base_ = '../../_base_/gdino_inc_base_coco.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'

dataset_type = 'ODVGIncDataset'
data_root = './data/coco/'
start = 0
end = 80

model = dict(
    type='GroundingDINO_inc',
    num_queries=900,
    backbone=dict(
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    bbox_head=dict(trunc_class=[start, end]),   #  only used in CdnQueryGenerator for training
    )

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/instances_train2017_od.json',
        label_map_file='annotations/coco2017_label_map.json',   
        start=start,
        end=end))

val_dataloader = dict(
    dataset=dict(
        start=start,
        end=end)) 

test_dataloader = val_dataloader

optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001))

# learning policy
max_epochs = 12

