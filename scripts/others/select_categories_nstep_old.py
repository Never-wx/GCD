# coding=utf-8
import os
import time
import json

def sel_cat_train(anno_file, total_phase):
    print('loading annotations into memory...')
    tic = time.time()
    dataset = json.load(open(anno_file, 'r'))
    assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
    print('Done (t={:0.2f}s)'.format(time.time()- tic))

    # sort by cat_ids
    dataset['categories'] = sorted(dataset['categories'], key=lambda k: k['id'])   
    num_cls = len(dataset['categories'])
    assert num_cls % total_phase == 0
    cls_per_phase = num_cls // total_phase
    for phase in range(total_phase):
        # select cats of cur phase, the cats is fixed
        start = phase * cls_per_phase
        end = (phase + 1) * cls_per_phase
        sel_cats = dataset['categories'][start:end]
        print(sel_cats)
        # selected annotations
        sel_cats_ids = [cat['id'] for cat in sel_cats]
        sel_anno = []
        sel_image_ids = []
        for anno in dataset['annotations']:
            if anno['category_id'] in sel_cats_ids:
                sel_anno.append(anno)
                sel_image_ids.append(anno['image_id'])
        # selected images
        sel_images = []
        for image in dataset['images']:
            if image['id'] in sel_image_ids:
                sel_images.append(image)
        # selected dataset
        sel_dataset = dict()
        sel_dataset['categories'] = sel_cats
        sel_dataset['annotations'] = sel_anno
        sel_dataset['images'] = sel_images
        # writing results
        fp = open(os.path.splitext(anno_file)[0] + '_%d-%d.json' % (start, end-1), 'w')
        json.dump(sel_dataset, fp)

def sel_cat_val(anno_file, total_phase):
    print('loading annotations into memory...')
    tic = time.time()
    dataset = json.load(open(anno_file, 'r'))
    assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
    print('Done (t={:0.2f}s)'.format(time.time()- tic))

    # sort by cat_ids
    dataset['categories'] = sorted(dataset['categories'], key=lambda k: k['id'])   
    num_cls = len(dataset['categories'])
    assert num_cls % total_phase == 0
    cls_per_phase = num_cls // total_phase
    for phase in range(total_phase):
        # select cats of cur phase, the cats is fixed
        start = 0
        end = (phase + 1) * cls_per_phase
        sel_cats = dataset['categories'][start:end]
        print(sel_cats)
        # selected annotations
        sel_cats_ids = [cat['id'] for cat in sel_cats]
        sel_anno = []
        sel_image_ids = []
        for anno in dataset['annotations']:
            if anno['category_id'] in sel_cats_ids:
                sel_anno.append(anno)
                sel_image_ids.append(anno['image_id'])
        # selected images
        sel_images = []
        for image in dataset['images']:
            if image['id'] in sel_image_ids:
                sel_images.append(image)
        # selected dataset
        sel_dataset = dict()
        # sel_dataset['categories'] = sel_cats              # BUG: cur cls, 70-79:[80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        sel_dataset['categories'] = dataset['categories']   # BUG: full cls, 0-79
        sel_dataset['annotations'] = sel_anno
        sel_dataset['images'] = sel_images
        # writing results
        fp = open(os.path.splitext(anno_file)[0] + '_%d-%d.json' % (start, end-1), 'w')
        json.dump(sel_dataset, fp)
        
if __name__ == "__main__":
    total_phase = 4
    anno_file_train = './data/coco/annotations/instances_train2017.json'
    sel_cat_train(anno_file_train, total_phase)
    anno_file_val = './data/coco/annotations/instances_val2017.json'
    sel_cat_val(anno_file_val, total_phase)
