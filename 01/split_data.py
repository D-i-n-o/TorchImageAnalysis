from sklearn.model_selection import train_test_split
import numpy as np
# import torch

import os
import shutil

NUM_CLASSES = 6
NUM_IMAGES = 17034

def verify_split(data_dir, output_dir):
    def __get_all_subset_images(subset):
        subset_images = []
        for cls in os.listdir(os.path.join(output_dir, subset)):
            cls_dir = os.path.join(output_dir, subset, cls)
            subset_images.extend(os.listdir(cls_dir))
        return subset_images
    
    def __is_split_disjoint(output_dir):
        train_images = __get_all_subset_images('train')
        val_images = __get_all_subset_images('val')
        test_images = __get_all_subset_images('test')
        return len(set(train_images).intersection(val_images)) == 0 and len(set(train_images).intersection(test_images)) == 0 and len(set(val_images).intersection(test_images)) == 0
    
    def __are_labels_preserved(data_dir, output_dir):
        original_labels_to_images = {}   
        for cls in os.listdir(data_dir):
            cls_dir = os.path.join(data_dir, cls)
            class_images = os.listdir(cls_dir)
            original_labels_to_images[cls] = frozenset(class_images) 

        for cls in os.listdir(data_dir):
            class_images = []
            for subset in ['train', 'val', 'test']:
                cls_dir = os.path.join(output_dir, subset, cls)
                class_images.extend(os.listdir(cls_dir))
            if set(class_images) != original_labels_to_images[cls]: return False
        return True
    
    # def __is_stratified(data_dir, output_dir):
    #     pass

    assert __is_split_disjoint(output_dir), 'The splits are not disjoint'
    assert __are_labels_preserved(data_dir, output_dir), 'The labels are not preserved'
    print(f'Split {data_dir} into {output_dir} is valid')
        
def copy_images(images, image_classes, data_dir, output_dir, subset_name):
    for i, img in enumerate(images):
        cls = image_classes[i]
        cls_dir = os.path.join(data_dir, cls)
        write_dir = os.path.join(output_dir, subset_name, cls)
        os.makedirs(write_dir, exist_ok=True)
        shutil.copy(os.path.join(cls_dir, img), write_dir)
        # print(f'Copying {cls}/{img} to {output_dir}/{subset_name}/{cls}')

def stratified_split(data_dir, output_dir, val_size, test_size):
    
    training_size = 1 - val_size - test_size

    classes = os.listdir(data_dir)
    images = []
    class_labels = []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        class_images = os.listdir(cls_dir)
        images.extend(class_images)
        class_labels.extend([cls] * len(class_images))
    
    train_val_images, test_images, train_val_classes, test_classes = train_test_split(images, class_labels, test_size=test_size, stratify=class_labels)
    train_images, val_images, train_classes, val_classes = train_test_split(train_val_images, train_val_classes, test_size=val_size, stratify=train_val_classes)
    copy_images(test_images, test_classes, data_dir, output_dir, 'test')
    copy_images(val_images, val_classes, data_dir, output_dir, 'val')
    copy_images(train_images, train_classes, data_dir, output_dir, 'train')
    
    
    

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join('/itf-fi-ml', 'shared', 'courses', 'IN3310', 'mandatory1_data')
    
    output_dir = os.path.join(script_dir, 'data')
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # test_set_percentage = 2000 / NUM_IMAGES
    # validation_set_percentage = 3000 / NUM_IMAGES
    # stratified_split(data_dir, output_dir, validation_set_percentage, test_set_percentage)
    
    verify_split(data_dir, output_dir)
    




