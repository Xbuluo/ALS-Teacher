#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import json
import os
import shutil

def split_img_vis_list(list_file, src_dir, out_dir):
    with open(list_file, 'r', encoding='utf-8') as f:
        file_list = json.load(f)
    all_files = dict()
    for file_ in glob.glob(os.path.join(src_dir, '*.png')):
        all_files[file_.split('/')[-1][-10:]] = file_
    print(f"Total images: {len(all_files)}")
    labeled_out_dir = out_dir['labeled']
    labeled_out_txt_dir = out_dir['labeled_txt']
    unlabeled_out_dir = out_dir['unlabeled']
    unlabeled_out_txt_dir = out_dir['unlabeled_txt']
    if os.path.exists(labeled_out_dir):
        shutil.rmtree(labeled_out_dir)
    if os.path.exists(unlabeled_out_dir):
        shutil.rmtree(unlabeled_out_dir)
    if os.path.exists(labeled_out_txt_dir):
        shutil.rmtree(labeled_out_txt_dir)
    if os.path.exists(unlabeled_out_txt_dir):
        shutil.rmtree(unlabeled_out_txt_dir)
    os.makedirs(labeled_out_dir)
    os.makedirs(unlabeled_out_dir)
    os.makedirs(labeled_out_txt_dir)
    os.makedirs(unlabeled_out_txt_dir)
    labeled_num = 0
    for file_name, file_path in all_files.items():
        # if '00' + file_name[1:] in file_list:
        if file_name in file_list:
            shutil.copyfile(file_path, os.path.join(labeled_out_dir, file_name))
            shutil.copyfile(file_path[:-16] + 'annfiles/' + file_name[:-4] + '.txt',
                            os.path.join(labeled_out_txt_dir, file_name[:-4] + '.txt'))
            labeled_num += 1
        else:
            shutil.copyfile(file_path, os.path.join(unlabeled_out_dir, file_name))
            shutil.copyfile(file_path[:-16] + 'annfiles/' + file_name[:-4] + '.txt',
                            os.path.join(unlabeled_out_txt_dir, file_name[:-4]+'.txt'))
    print(f"Finish saving {labeled_num} labeled image.")


if __name__ == '__main__':
    # example
    # dota dataset path
    dota_root = os.path.join(os.getcwd(), "../../../data/dota")
    # list file
    list_file = os.path.join(os.getcwd(), "../../../data_lists/10p_list.json")
    src_dir = os.path.join(dota_root, 'train/images')
    out_dir = dict(
        labeled=f'{dota_root}/train_10_labeled/images',
        unlabeled=f'{dota_root}/train_10_unlabeled/images',
        labeled_txt = f'{dota_root}/train_10_labeled/annfiles',
        unlabeled_txt = f'{dota_root}/train_10_unlabeled/annfiles',
    )
    split_img_vis_list(list_file, src_dir, out_dir)
