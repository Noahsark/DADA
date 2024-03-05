
import numpy as np, os, pandas as pd

from datasets.basic_dataset import BaseDataset


def give_dataloaders(opt, datapath=None, splitpath=None):
        
    train_ys, train_im_paths = [], []
    query_ys, query_im_paths = [], []
    gallery_ys, gallery_im_paths = [], []
    
    data_info = np.array(pd.read_table(opt.source_path +'/list_eval_partition.txt', 
        header=1, delim_whitespace=True))[:,:]
    #Separate into training dataset and query/gallery dataset for testing.
    train = data_info[data_info[:,2]=='train'][:,:2]
    query = data_info[data_info[:,2]=='query'][:,:2]
    gallery = data_info[data_info[:,2]=='gallery'][:,:2]
                            
                            
    #Generate conversions
    lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in train[:,1]])))}
    train[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in train[:,1]])
    lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) 
                    for x in np.concatenate([query[:,1], gallery[:,1]])])))}
    query[:,1]   = np.array([lab_conv[int(x.split('_')[-1])] for x in query[:,1]])
    gallery[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in gallery[:,1]])
    
    
    train_image_dict = {}
    for img_path, key in train:
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(os.path.join(opt.source_path, img_path))
        
    query_image_dict = {}
    for img_path, key in query:
        if not key in query_image_dict.keys():
            query_image_dict[key] = []
        query_image_dict[key].append(os.path.join(opt.source_path, img_path))
    
    gallery_image_dict = {}
    for img_path, key in gallery:
        if not key in gallery_image_dict.keys():
            gallery_image_dict[key] = []
        gallery_image_dict[key].append(os.path.join(opt.source_path, img_path))

    train_classes = sorted(list(train_image_dict.keys()))
    query_classes = sorted(list(query_image_dict.keys()))
    gallery_classes = sorted(list(gallery_image_dict.keys()))
                                     
    train_conversion = {
        i: classname
        for i, classname in enumerate(train_classes)
    }
    query_conversion = {
        i: classname
        for i, classname in enumerate(query_classes)
    }
    gallery_conversion = {
        i: classname
        for i, classname in enumerate(gallery_classes)
    }
    
    # sort
    train_image_dict = {
        i: train_image_dict[key]
        for i, key in enumerate(train_classes)
    }
    query_image_dict = {
        i: query_image_dict[key]
        for i, key in enumerate(query_classes)
    }
    gallery_image_dict = {
        i: gallery_image_dict[key]
        for i, key in enumerate(gallery_classes)
    }
    
    print(
        '\nDataset Setup: \n#Classes: Train ({0}) | Query ({1}) | Gallery ({2})\n'.
        format(len(train_image_dict),
               len(query_image_dict),
               len(gallery_image_dict)))
    
    train_dataset = BaseDataset(train_image_dict, opt)
    query_dataset = BaseDataset(query_image_dict, opt, is_validation=True)
    gallery_dataset = BaseDataset(gallery_image_dict, opt, is_validation=True)

    train_dataset.conversion = train_conversion
    query_dataset.conversion = query_conversion
    gallery_dataset.conversion = gallery_conversion
    
    return {
        'training': train_dataset,
        'query': query_dataset,
        'gallery': gallery_dataset,
    }
    
