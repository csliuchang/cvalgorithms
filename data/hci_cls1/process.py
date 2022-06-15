#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-08-30 18:46
# @Project : deepcv_examples

import os


def create_cls_txt():
    img_files = os.listdir("images")
    train_val_ratio = 0.7

    good_img_files = [img_file for img_file in img_files
                      if not os.path.exists(os.path.join("labelme", img_file.replace(".PNG", ".json")))]
    bad_img_files = [img_file for img_file in img_files
                     if os.path.exists(os.path.join("labelme", img_file.replace(".PNG", ".json")))]

    num_of_good = len(good_img_files)
    num_of_bad = len(bad_img_files)

    train_img_files = [good_img_files[:int(num_of_good * train_val_ratio)],
                       bad_img_files[:int(num_of_bad * train_val_ratio)]]

    val_img_files = [good_img_files[int(num_of_good * train_val_ratio):],
                     bad_img_files[int(num_of_bad * train_val_ratio):]]

    split_files = {
        "train": train_img_files,
        "val": val_img_files
    }

    for s, files in split_files.items():
        fp = open("cls_{}.txt".format(s), "w")
        good_files = files[0]
        bad_files = files[1]
        
        for f in good_files:
            line = "images/{}\tgood".format(f)
            fp.write(line + "\n")
            
        for f in bad_files:
            line = "images/{}\tbad".format(f)
            fp.write(line + "\n")

        fp.close()


def create_labelme_txt():
    json_files = os.listdir("labelme")
    
    num_of_files = len(json_files)
    train_val_ratio = 0.7
    
    split_files = {
        "train": json_files[:int(num_of_files * train_val_ratio)],
        "val": json_files[int(num_of_files * train_val_ratio):]
    }
    
    for s, files in split_files.items():
        fp = open("labelme_{}.txt".format(s), "w")
        for f in files:
            line = "images/{}\tlabelme/{}".format(f.replace(".json", ".PNG"), f)
            fp.write(line + "\n")
        fp.close()
        pass
    
    pass


if __name__ == "__main__":
    create_cls_txt()
    create_labelme_txt()
