# coding=utf-8
import os
import time
import json

def sel_cat_train(anno_file, sel_num, output_dir):
    print('Loading annotations into memory...')
    tic = time.time()
    
    # Load the dataset
    dataset = json.load(open(anno_file, 'r'))
    assert isinstance(dataset, dict), 'Annotation file format {} not supported'.format(type(dataset))
    print('Done (t={:0.2f}s)'.format(time.time() - tic))

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Sort categories by ID
    dataset['categories'] = sorted(dataset['categories'], key=lambda k: k['id'])

    # Select the first sel_num categories
    sel_cats = dataset['categories'][:sel_num]
    sel_cats_ids = [cat['id'] for cat in sel_cats]

    # Filter annotations and images for the selected categories
    sel_anno, sel_image_ids = [], []
    for anno in dataset['annotations']:
        if anno['category_id'] in sel_cats_ids:
            sel_anno.append(anno)
            sel_image_ids.append(anno['image_id'])

    sel_images = [image for image in dataset['images'] if image['id'] in sel_image_ids]
    
    # Save the first part of the dataset
    sel_dataset = {'categories': sel_cats, 'annotations': sel_anno, 'images': sel_images}
    output_path = os.path.join(output_dir, os.path.basename(anno_file).replace('.json', '_0-{}.json'.format(sel_num - 1)))
    with open(output_path, 'w') as fp:
        json.dump(sel_dataset, fp)
    print(f"Saved {output_path}")

    # Select the remaining categories
    sel_cats = dataset['categories'][sel_num:]
    sel_cats_ids = [cat['id'] for cat in sel_cats]

    # Filter annotations and images for the remaining categories
    sel_anno, sel_image_ids = [], []
    for anno in dataset['annotations']:
        if anno['category_id'] in sel_cats_ids:
            sel_anno.append(anno)
            sel_image_ids.append(anno['image_id'])

    sel_images = [image for image in dataset['images'] if image['id'] in sel_image_ids]

    # Ensure the full category list is retained
    sel_dataset = {'categories': sel_cats, 'annotations': sel_anno, 'images': sel_images}
    output_path = os.path.join(output_dir, os.path.basename(anno_file).replace('.json', '_{}-79.json'.format(sel_num)))
    with open(output_path, 'w') as fp:
        json.dump(sel_dataset, fp)
    print(f"Saved {output_path}")

def sel_cat_val(anno_file, sel_num, output_dir):
    print('Loading annotations into memory...')
    tic = time.time()
    
    # Load the dataset
    dataset = json.load(open(anno_file, 'r'))
    assert isinstance(dataset, dict), 'Annotation file format {} not supported'.format(type(dataset))
    print('Done (t={:0.2f}s)'.format(time.time() - tic))

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Sort categories by ID
    dataset['categories'] = sorted(dataset['categories'], key=lambda k: k['id'])

    # Select the first sel_num categories
    sel_cats = dataset['categories'][:sel_num]
    sel_cats_ids = [cat['id'] for cat in sel_cats]

    # Filter annotations and images for the selected categories
    sel_anno, sel_image_ids = [], []
    for anno in dataset['annotations']:
        if anno['category_id'] in sel_cats_ids:
            sel_anno.append(anno)
            sel_image_ids.append(anno['image_id'])

    sel_images = [image for image in dataset['images'] if image['id'] in sel_image_ids]
    
    # Save the first part of the dataset
    sel_dataset = {'categories': sel_cats, 'annotations': sel_anno, 'images': sel_images}
    output_path = os.path.join(output_dir, os.path.basename(anno_file).replace('.json', '_0-{}.json'.format(sel_num - 1)))
    with open(output_path, 'w') as fp:
        json.dump(sel_dataset, fp)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    sel_num = 40
    output_dir = './data/coco/annotations/40+40/'
    
    anno_file_train = './data/coco/annotations/instances_train2017.json'
    sel_cat_train(anno_file_train, sel_num, output_dir)

    anno_file_val = './data/coco/annotations/instances_val2017.json'
    sel_cat_val(anno_file_val, sel_num, output_dir)
