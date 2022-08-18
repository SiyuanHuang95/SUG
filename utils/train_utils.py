import random

import torch
import torch.nn.functional as F


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None
    version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def adjust_learning_rate(optimizer, epoch, lr, scaler, writer):
    """Sets the learning rate to the initial LR decayed by half by every 5 or 10 epochs"""
    if epoch > 0:
        if epoch <= 30:
            lr = lr * scaler * (0.5 ** (epoch // 5))
        else:
            lr = lr * scaler * (0.5 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        writer.add_scalar('lr_dis', lr, epoch)


def discrepancy(out1, out2):
    """discrepancy loss"""
    out = torch.mean(torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))
    return out


def make_variable(tensor, volatile=False):
    from torch.autograd import Variable
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


class Sampler():
    def __init__(self, classes, class_per_batch, batch_size):
        self.classes = classes
        self.n_batches = sum([len(x) for x in classes]) // batch_size
        self.class_per_batch = class_per_batch
        self.batch_size = batch_size

    def __iter__(self):
        classes = random.sample(range(len(self.classes)), self.class_per_batch)
        
        batches = []
        for _ in range(self.n_batches):
            batch = []
            for i in range(self.batch_size):
                klass = random.choice(classes)
                batch.append(random.choice(self.classes[klass]))
            batches.append(batch)
        return iter(batches)