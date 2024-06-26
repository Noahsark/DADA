import copy
import os

from datasets.basic_dataset import BaseDataset


def give_info_dict(source, classes):
    image_list = [[(i, source + '/' + classname + '/' + x)
                   for x in sorted(os.listdir(source + '/' + classname))
                   if '._' not in x] for i, classname in enumerate(classes)]
    image_list = [x for y in image_list for x in y]

    idx_to_class_conversion = {
        i: classname
        for i, classname in enumerate(classes)
    }

    image_dict = {}
    for key, img_path in image_list:
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)

    return image_list, image_dict, idx_to_class_conversion


def give_dataloaders(opt, datapath):
    image_sourcepath = datapath + '/images'
    image_classes = sorted([x for x in os.listdir(image_sourcepath)])
    total_conversion = {i: x for i, x in enumerate(image_classes)}
    image_list = {
        i: sorted([
            image_sourcepath + '/' + key + '/' + x
            for x in os.listdir(image_sourcepath + '/' + key)
        ])
        for i, key in enumerate(image_classes)
    }
    image_list = [[(key, img_path) for img_path in image_list[key]]
                  for key in image_list.keys()]
    image_list = [x for y in image_list for x in y]

    train_classes, test_classes = image_classes[:len(image_classes) //
                                                2], image_classes[
                                                    len(image_classes) // 2:]

    ###
    
    val_dataset = None

    train_image_list, train_image_dict, train_conversion = give_info_dict(
        image_sourcepath, train_classes)
    test_image_list, test_image_dict, test_conversion = give_info_dict(
        image_sourcepath, test_classes)

    ###
    print(
        '\nDataset Setup: \n#Classes: Train ({0}) | Test ({1})\n'.
        format(len(train_image_dict),
               len(test_image_dict)))
    
    ###
    train_dataset = BaseDataset(train_image_dict, opt)
    test_dataset = BaseDataset(test_image_dict, opt, is_validation=True)
    train_eval_dataset = BaseDataset(train_image_dict, opt, is_validation=True)

    ###
    reverse_train_conversion = {
        item: key
        for key, item in train_conversion.items()
    }
    reverse_test_conversion = {
        item: key
        for key, item in test_conversion.items()
    }

    train_dataset.conversion = train_conversion
    test_dataset.conversion = test_conversion
    train_eval_dataset.conversion = train_conversion

    train_language_conversion = {
        key: item.split('.')[-1].replace('_', ' ')
        for key, item in train_conversion.items()
    }
    test_language_conversion = {
        key: item.split('.')[-1].replace('_', ' ')
        for key, item in test_conversion.items()
    }

    train_dataset.language_conversion = train_language_conversion
    test_dataset.language_conversion = test_language_conversion
    train_eval_dataset.language_conversion = train_language_conversion

    return {
        'training': train_dataset,
        'validation': val_dataset,
        'testing': test_dataset,
        'evaluation': train_eval_dataset
    }
