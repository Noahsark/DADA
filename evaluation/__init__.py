
import numpy as np
import torch
QUERY_GALLERY_RECALL_KS = [1, 10, 20, 30, 40, 50]

def evaluate(dataset,
             LOG,
             metric_computer,
             dataloaders,
             model,
             opt,
             evaltypes,
             device,
             aux_store=None,
             log_key='Test',
             compute_metrics_only=False,
             print_text=True, criterion=None):
    """
    Parent-Function to compute evaluation metrics, print summary string and
    store checkpoint files/plot sample recall plots.
    """
    if not opt.dataset == 'inshop':
        computed_metrics, extra_infos = metric_computer.compute_standard(
            opt, model, dataloaders[0], evaltypes, device, mode=log_key)
        
        if opt.save_results and opt.epoch % opt.save_step == 0:
            savepath = LOG.prop.save_path + '/extra_info_{}.npy'.format(opt.epoch)
            np.save(savepath, extra_infos['embeds'])
            print ('save extra_infos to {} \n'.format(savepath))
        
        numeric_metrics = {}
        histogr_metrics = {}
        for main_key in computed_metrics.keys():
            for name, value in computed_metrics[main_key].items():
                if isinstance(value, np.ndarray):
                    if main_key not in histogr_metrics:
                        histogr_metrics[main_key] = {}
                    histogr_metrics[main_key][name] = value
                else:
                    if main_key not in numeric_metrics:
                        numeric_metrics[main_key] = {}
                    numeric_metrics[main_key][name] = value
    else:
        # inshop
        recalls, map = metric_computer.evaluate_query_gallery(model, 
                    dataloaders['query'], dataloaders['gallery'],
                    QUERY_GALLERY_RECALL_KS)
        numeric_metrics = {}
        numeric_metrics['embeds'] = {}

        for k, recall in zip(QUERY_GALLERY_RECALL_KS, recalls):
            key = "e_recall@{}".format(k)
            numeric_metrics['embeds'][key] = recall
        numeric_metrics['embeds']['MAP'] = map
    
    ###
    full_result_str = ''
    for evaltype in numeric_metrics.keys():
        full_result_str += 'Embed-Type: {}:\n'.format(evaltype)
        for i,(metricname, metricval) in enumerate(numeric_metrics[evaltype].items()):
            full_result_str += '{0}{1}: {2:4.4f}'.format('\n' if i>0 else '',metricname, metricval)
        full_result_str += '\n'

    if print_text:
        print(full_result_str)
    
    if not compute_metrics_only:
        
        for evaltype in numeric_metrics.keys():
            for eval_metric in numeric_metrics[evaltype].keys():
                parent_metric = evaltype + '_{}'.format(
                    eval_metric.split('@')[0])
                LOG.progress_saver[log_key].log(
                    eval_metric,
                    numeric_metrics[evaltype][eval_metric],
                    group=parent_metric)
        
        ###
        for evaltype in evaltypes:
            for storage_metric in opt.storage_metrics:
                parent_metric = evaltype + '_{}'.format(
                    storage_metric.split('@')[0])
                ref_mets = LOG.progress_saver[log_key].groups[parent_metric][
                    storage_metric]['content'][:-1]
                if not len(ref_mets): ref_mets = [-np.inf]

                if numeric_metrics[evaltype][storage_metric] > np.max(
                        ref_mets) and opt.store_improvements:
                    print('Saved improved checkpoint for {}: {}\n'.format(
                        log_key, storage_metric))
                    set_checkpoint(model,
                                   opt,
                                   LOG,
                                   LOG.prop.save_path +
                                   '/checkpoint_{}_{}_{}.pth.tar'.format(
                                       log_key, evaltype, storage_metric),
                                   aux=aux_store, criterion=criterion)
    else:
        return numeric_metrics

def set_checkpoint(model, opt, progress_saver, savepath, criterion=None, aux=None):
    if criterion:
        # save criterion.proxies, criterion.saved_batch_mix
        if criterion.saved_batch_mix == None:
            criterion.saved_batch_mix = torch.empty(0)
            print('Saved proxies \n')
        else:
            if criterion.saved_batch_mix.shape[0] > 0:
                print('Saved proxies and mixing features \n')
            else:
                print('Saved proxies \n')
        torch.save(
            {
                'state_dict': model.state_dict(),
                'proxy':    criterion.proxies.detach(),
                'saved_batch_mix': criterion.saved_batch_mix.detach(),
                'opt': opt,
                'progress': progress_saver,
                'aux': aux
            }, savepath)
    else:
        torch.save(
            {
                'state_dict': model.state_dict(),
                'opt': opt,
                'progress': progress_saver,
                'aux': aux
            }, savepath)


def save_numpy(model, opt, savepath, criterion):
    print('saving {}...\n'.format(savepath))
    criterion.save_average()
    np.savez_compressed(savepath, source=criterion.s_predictions, 
            target=criterion.t_predictions, sims=criterion.proxy_sims)
    