import numpy as np
import torch
import tqdm

import torch.nn.functional as F
from tqdm import tqdm


def evaluate_cos_Inshop(model, query_dataloader, gallery_dataloader):
    query_X, query_T = predict_batchwise(model, query_dataloader, name='Query')
    gallery_X, gallery_T = predict_batchwise(model, gallery_dataloader, name='Gallery')
    
    query_X = F.normalize(query_X, dim=-1)
    gallery_X = F.normalize(gallery_X, dim=-1)
    
    cos_sim = F.linear(query_X, gallery_X)
    
    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0
        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]
            thresh = torch.max(pos_sim).item()
            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1
        return match_counter / m
    
    recall = []
    keys = []
    for k in [1, 10, 20, 30, 40, 50]:
        r_at_k = recall_k(cos_sim, query_T, gallery_T, k)
        recall.append(r_at_k)
        keys.append("e_recall@{}".format(k))
                
    return recall, keys



def predict_batchwise(model, dataloader, name):

    data_iterator = tqdm(dataloader,
                bar_format='{desc}{percentage:2.2f}% [{n_fmt}/{total_fmt}, '
                           '{elapsed}/{remaining}{postfix}]',
                ncols=96, ascii=True, desc='[Embedding {}]: '.format(name))
    
    ds = dataloader.dataset
    
    embs = []
    targets = []
    with torch.no_grad():
        for i, out in enumerate(data_iterator):
            class_labels, input_dict, sample_indices = out
            input = input_dict['image']
            emb = model(input.cuda())['embeds']
            for j in emb:
                embs.append(j)
            for j in class_labels:
                targets.append(j)

    return torch.stack(embs), torch.stack(targets)
