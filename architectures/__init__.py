
import architectures.resnet50 as resnet50
import architectures.bninception as bninception

def select(arch, opt):

    if 'resnet50' in arch:
        return resnet50.Network(opt)
    if 'bninception' in arch:
        return bninception.Network(opt)

