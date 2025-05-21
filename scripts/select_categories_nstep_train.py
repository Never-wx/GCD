# coding=utf-8
import os
import time
import json

def sel_cat_train(anno_file_path, total_phase, output_dir):
    """Splits training dataset categories into phases and saves them in output directory."""
    print('Loading annotations into memory...')
    tic = time.time()

    # Load dataset
    dataset = json.load(open(anno_file_path, 'r'))
    assert isinstance(dataset, dict), 'Annotation file format {} not supported'.format(type(dataset))
    print('Done (t={:0.2f}s)'.format(time.time() - tic))
    
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Sort categories by ID
    dataset['categories'] = sorted(dataset['categories'], key=lambda k: k['id'])
    num_cls = len(dataset['categories'])
    
    assert num_cls % total_phase == 0, "Total categories must be evenly divisible by total phases."
    cls_per_phase = num_cls // total_phase

    for phase in range(total_phase):
        # Select categories for the current phase
        start = phase * cls_per_phase
        end = (phase + 1) * cls_per_phase
        sel_cats = dataset['categories'][start:end]

        # Select corresponding annotations and images
        sel_cats_ids = [cat['id'] for cat in sel_cats]
        sel_anno, sel_image_ids = [], []
        for anno in dataset['annotations']:
            if anno['category_id'] in sel_cats_ids:
                sel_anno.append(anno)
                sel_image_ids.append(anno['image_id'])

        sel_images = [image for image in dataset['images'] if image['id'] in sel_image_ids]

        # Create the selected dataset
        sel_dataset = {'categories': sel_cats, 'annotations': sel_anno, 'images': sel_images}

        # Save the split dataset
        anno_file = anno_file_path.split('/')[-1]
        start = int(os.path.basename(anno_file).split('_')[-1].split('-')[0]) + start
        end = int(os.path.basename(anno_file).split('_')[-1].split('-')[0]) + end

        output_path = os.path.join(output_dir, 'instances_train2017_' + '{}-{}.json'.format(start, end - 1))
        with open(output_path, 'w') as fp:
            json.dump(sel_dataset, fp)
        print(f"Saved {output_path}")

if __name__ == "__main__":
    total_phase = 4
    output_dir = './data/coco/annotations/40+10_4/test'
    anno_file_train = './data/coco/annotations/40+40/instances_train2017_40-79.json'
    sel_cat_train(anno_file_train, total_phase, output_dir)