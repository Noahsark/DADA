import datasets.cars196 as cars196
import datasets.cub200 as cub200
import datasets.stanford_online_products as stanford_online_products
import datasets.inshop as inshop

def select(dataset, opt, data_path, splitpath=None):
    if 'cub200' in dataset:
        return cub200.give_dataloaders(opt, data_path)
    elif 'cars196' in dataset:
        return cars196.give_dataloaders(opt, data_path)
    elif 'online_products' in dataset:
        return stanford_online_products.give_dataloaders(opt, data_path)
    elif 'inshop' in dataset:
        return inshop.give_dataloaders(opt, data_path)
    else:
        raise NotImplementedError(
            'A dataset for {} is currently not implemented.\n\Currently available are : cub200, cars196 and online_products!'
            .format(dataset))
