import copy
import os

import numpy as np
import pandas as pd
import pdb

from datasets.basic_dataset import BaseDataset


def give_dataloaders(opt, datapath=None, splitpath=None):
    image_sourcepath = opt.source_path + '/images'
    training_files = pd.read_table(opt.source_path +
                                   '/Info_Files/Ebay_train.txt',
                                   header=0,
                                   delimiter=' ')
    test_files = pd.read_table(opt.source_path + '/Info_Files/Ebay_test.txt',
                               header=0,
                               delimiter=' ')

    train_classes = sorted(
        np.unique([
            x.split('/')[-1].split('_')[0] for x in training_files['path']
        ]).tolist())
    test_classes = sorted(
        np.unique([x.split('/')[-1].split('_')[0]
                   for x in test_files['path']]).tolist())

    super_dict = {}
    class_dict = {}
    class_conversion = {}

    for i, (super_ix, class_ix, image_path) in enumerate(
            zip(training_files['super_class_id'], training_files['class_id'],
                training_files['path'])):
        if super_ix not in super_dict: super_dict[super_ix] = {}
        if class_ix not in super_dict[super_ix]:
            super_dict[super_ix][class_ix] = []
        super_dict[super_ix][class_ix].append(image_sourcepath + '/' +
                                              image_path)

        if class_ix not in class_dict: class_dict[class_ix] = []
        class_dict[class_ix].append(image_sourcepath + '/' + image_path)
        class_conversion[class_ix] = image_path.split('/')[-1].split('_')[0]

    for i, (super_ix, class_ix, image_path) in enumerate(
            zip(test_files['super_class_id'], test_files['class_id'],
                test_files['path'])):
        if super_ix not in super_dict: super_dict[super_ix] = {}
        if class_ix not in super_dict[super_ix]:
            super_dict[super_ix][class_ix] = []
        super_dict[super_ix][class_ix].append(image_sourcepath + '/' +
                                              image_path)
        
        if class_ix not in class_dict: class_dict[class_ix] = []
        class_dict[class_ix].append(image_sourcepath + '/' + image_path)
        class_conversion[class_ix] = image_path.split('/')[-1].split('_')[0]
    
    train_image_dict = {
        key: item
        for key, item in class_dict.items()
        if str(class_conversion[key]) in train_classes
    }

    test_image_dict = {
        key: item
        for key, item in class_dict.items()
        if str(class_conversion[key]) in test_classes
    }

    train_classes, test_classes = sorted(list(
        train_image_dict.keys())), sorted(list(test_image_dict.keys()))
    train_conversion = {
        i: classname
        for i, classname in enumerate(train_classes)
    }
    test_conversion = {
        i: classname
        for i, classname in enumerate(test_classes)
    }

    train_image_dict = {
        i: train_image_dict[key]
        for i, key in enumerate(train_classes)
    }
    test_image_dict = {
        i: test_image_dict[key]
        for i, key in enumerate(test_classes)
    }

    ##

    val_dataset = None

    print(
        '\nDataset Setup: \n#Classes: Train ({0}) | Test ({1})\n'.
        format(len(train_image_dict),
               len(test_image_dict)))

    train_dataset = BaseDataset(train_image_dict, opt)
    eval_dataset = BaseDataset(train_image_dict, opt, is_validation=True)
    test_dataset = BaseDataset(test_image_dict, opt, is_validation=True)

    reverse_train_conversion = {
        item: key
        for key, item in train_conversion.items()
    }
    reverse_test_conversion = {
        item: key
        for key, item in test_conversion.items()
    }

    train_dataset.conversion = train_conversion
    eval_dataset.conversion = train_conversion
    test_dataset.conversion = test_conversion

    train_language_conversion = {}
    train_conversion_list = []
    for key, subdict in train_image_dict.items():
        super_class = subdict[0].split('/')[-2].replace('_final',
                                                        '').replace('_', ' ')
        train_language_conversion[key] = super_class
        if super_class not in train_conversion_list:
            train_conversion_list.append(super_class)

    test_language_conversion = {}
    test_conversion_list = []
    for key, subdict in test_image_dict.items():
        super_class = subdict[0].split('/')[-2].replace('_final',
                                                        '').replace('_', ' ')
        test_language_conversion[key] = super_class
        if super_class not in test_conversion_list:
            test_conversion_list.append(super_class)

    train_dataset.language_conversion = train_language_conversion
    test_dataset.language_conversion = test_language_conversion
    eval_dataset.language_conversion = train_language_conversion

    return {
        'training': train_dataset,
        'validation': val_dataset,
        'testing': test_dataset,
        'evaluation': eval_dataset,
        'evaluation_train': None,
        'super_evaluation': None
    }
