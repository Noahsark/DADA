
import importlib
import os
import criteria


def select(loss, opt, to_optim=None):
    losses = os.listdir(os.path.join(os.getcwd(), 'criteria'))
    losses = [x.split('.py')[0] for x in losses if '__' not in x]
    losses = {x: 'criteria.{}'.format(x) for x in losses}

    if loss not in losses:
        raise NotImplementedError('Loss {} not implemented!'.format(loss))

    loss_lib = importlib.import_module(format(losses[loss]))

    loss_par_dict = {'opt': opt}

    criterion = loss_lib.Criterion(**loss_par_dict)
    
    # update: will not select any par if no optim_dict_list
    if to_optim is not None:
        
        if hasattr(criterion, 'optim_dict_list'
                    ) and criterion.optim_dict_list is not None:
            to_optim += criterion.optim_dict_list
                
        return criterion, to_optim
    else:
        return criterion
