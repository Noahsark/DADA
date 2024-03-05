import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def rank_gallery(scores, gallery_labels, k):
    """
    Get top-K gallery labels based on the scores.
    """
    top_k_indices = np.argpartition(-scores, k)[:k]
    top_k_indices_sorted = top_k_indices[np.argsort(-scores[top_k_indices])] # Sort these top K
    return gallery_labels[top_k_indices_sorted]


def compute_recalls_for_ks(query_label, ranked_labels, ks):
    """
    Compute recalls for all specified k-values.
    """
    recalls = []
    hit = 0  # Binary indicator to denote if a relevant image has been encountered within top-k

    for i, label in enumerate(ranked_labels):
        if label == query_label:
            hit = 1
        
        if (i+1) in ks:
            recalls.append(hit)
    
    return recalls



def average_precision(query_label, ranked_labels):
    """
    Calculate Average Precision for a single query.
    """
    relevant_items = 0
    cumulated_precision = 0.0
    for i, label in enumerate(ranked_labels):
        if label == query_label:
            relevant_items += 1
            precision_at_i = relevant_items / (i + 1.0)
            cumulated_precision += precision_at_i
    return cumulated_precision / relevant_items if relevant_items > 0 else 0.0


def average_precision_at_k(query_label, ranked_labels, k=1000):
    """
    Calculate Average Precision for a single query considering the first k ranked labels.
    """
    relevant_items = 0
    cumulated_precision = 0.0
    for i, label in enumerate(ranked_labels[:k]):
        if label == query_label:
            relevant_items += 1
            precision_at_i = relevant_items / (i + 1.0)
            cumulated_precision += precision_at_i
    return cumulated_precision / relevant_items if relevant_items > 0 else 0.0


def MAP_at_k(scores, query_labels, gallery_labels, k=1000):
    """
    Calculate Mean Average Precision (MAP) for given queries and gallery considering the first k ranked samples.
    """
    
    ranked_labels_list = [rank_gallery(score, gallery_labels, k) for score in scores]
    
    all_aps = [average_precision_at_k(q_label, ranked_labels, k) for q_label, ranked_labels in zip(query_labels, ranked_labels_list)]
    
    return np.mean(all_aps)


# refer embeddings
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


def evaluate_query_gallery(model, query_dataloader, gallery_dataloader, ks):
    query_X, query_labels = predict_batchwise(model, query_dataloader, name='Query')
    gallery_X, gallery_labels = predict_batchwise(model, gallery_dataloader, name='Gallery')
    
    ks = sorted(ks)
    
    query_X = F.normalize(query_X, dim=-1)
    gallery_X = F.normalize(gallery_X, dim=-1)
    scores = F.linear(query_X, gallery_X).cpu().numpy()
    max_k = max(ks)
    
    ranked_labels_list = [rank_gallery(score, gallery_labels, max_k) for score in scores]
    all_recalls = []
    for q_label, ranked_labels in zip(query_labels, ranked_labels_list):
        recalls = compute_recalls_for_ks(q_label, ranked_labels, ks)
        all_recalls.append(recalls)
    
    mean_recalls = np.mean(all_recalls, axis=0)
    
    aps = [average_precision(q_label, ranked_labels) for q_label, ranked_labels in zip(query_labels, ranked_labels_list)]
    
    map = np.mean(aps)

    return mean_recalls, map