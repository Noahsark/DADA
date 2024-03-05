import torch
import numpy as np
import faiss


# in all case we assume R < 1000 for efficiency
# MAP@R: https://arxiv.org/abs/2003.08505

class Metric():
    def __init__(self, **kwargs):
        self.requires = ['features', 'target_labels']
        self.name     = 'mAP_R'

    def __call__(self, target_labels, features):
        labels, freqs = np.unique(target_labels, return_counts=True)
        max_R = int(min(max(freqs + 1), 1000))

        faiss_search_index  = faiss.IndexFlatL2(features.shape[-1])
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
            res = faiss.StandardGpuResources()
            faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
                    
        faiss_search_index.add(features)
        nearest_neighbours  = faiss_search_index.search(features, max_R)[1][:,1:]
        
        target_labels = target_labels.reshape(-1)
        nn_labels = target_labels[nearest_neighbours]

        avg_r_precisions = []
        for label, freq in zip(labels, freqs):
            rows_with_label = np.where(target_labels==label)[0] # return x, y
            
            # for each query
            for row in rows_with_label:
                r_recalled_samples           = np.arange(1, freq + 1)
                target_label_occ_in_row      = (nn_labels[row,:]==label)[: freq]
                cumsum_target_label_freq_row = np.cumsum(target_label_occ_in_row)
                avg_r_pr_row = np.sum(cumsum_target_label_freq_row*target_label_occ_in_row/r_recalled_samples)/freq
                avg_r_precisions.append(avg_r_pr_row)

        return np.mean(avg_r_precisions)