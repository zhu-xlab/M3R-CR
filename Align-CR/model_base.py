import torch
import os
import numpy as np
from torch.optim import lr_scheduler
import random

class ModelBase():
    def save_network(self, network, optimizer, epoch, lr_scheduler, save_dir):
        checkpoint = {
            "network": network.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
            "lr_scheduler": lr_scheduler.state_dict()
            }
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_filename = '%s_net_CR.pth' % (str(epoch))
        save_path = os.path.join(save_dir, save_filename)
        torch.save(checkpoint, save_path)

    def print_networks(self, network):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in network.parameters():
            num_params += param.numel()
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    def update_lr(self):
        self.lr_scheduler.step()
        for param_group in self.optimizer_G.param_groups:
            print('optimizer_G_lr', param_group['lr'])

def print_options(opts):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opts).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    if not os.path.exists(opts.save_model_dir):
        os.makedirs(opts.save_model_dir)
    file_name = os.path.join(opts.save_model_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True