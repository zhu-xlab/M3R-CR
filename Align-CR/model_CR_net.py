import os
import torch
import torch.nn as nn

from model_base import ModelBase

from net_CR_RDN import RDN_residual_CR

from metrics import PSNR, SSIM, SAM, MAE

from torch.optim import lr_scheduler

from losses import ECCharbonnierLoss

class ModelCRNet(ModelBase):
    def __init__(self, opts):
        super(ModelCRNet, self).__init__()
        self.opts = opts
        
        # create network
        self.net_G = RDN_residual_CR(self.opts.crop_size).cuda()
        self.net_G = nn.DataParallel(self.net_G)
        self.print_networks(self.net_G)

        # initialize optimizers
        SFAlign_params = list(map(id, self.net_G.module.SFAlign.parameters())) 
        base_params = filter(lambda x:id(x) not in SFAlign_params, self.net_G.parameters())  
        if self.opts.optimizer == 'SGD':
            self.optimizer_G = torch.optim.SGD([
                {'params': base_params, 'lr': self.opts.lr},
                {'params': self.net_G.module.SFAlign.parameters(), 'lr': self.opts.lr * 0.1}], momentum=0.9, weight_decay=1e-4)
        elif self.opts.optimizer == 'Adam':
            self.optimizer_G = torch.optim.Adam([
                {'params': base_params, 'lr': self.opts.lr},
                {'params': self.net_G.module.SFAlign.parameters(), 'lr': self.opts.lr * 0.1}])
            
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=self.opts.lr_step, gamma=0.5)
        
        self.loss_fn=ECCharbonnierLoss()
                        
    def set_input(self, inputs):
        self.cloudy_data = inputs['cloudy_data'].cuda()
        self.cloudfree_data = inputs['cloudfree_data'].cuda()
        self.SAR_data = inputs['SAR_data'].cuda()
        self.cloudmask_data = inputs['cloudmask_data'].cuda()
        
    def forward(self):
        pred_cloudfree_data = self.net_G(self.cloudy_data, self.SAR_data)
        return pred_cloudfree_data

    def optimize_parameters(self):              
        self.pred_cloudfree_data = self.forward()

        self.loss_G = self.loss_fn(self.pred_cloudfree_data, self.cloudfree_data, self.cloudmask_data)

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()  

        return self.loss_G.item()

    def val_scores(self):
        self.pred_cloudfree_data = self.forward()
        scores = {'PSNR': PSNR(self.pred_cloudfree_data.data, self.cloudfree_data),
                  'SSIM': SSIM(self.pred_cloudfree_data.data, self.cloudfree_data),
                  'SAM': SAM(self.pred_cloudfree_data.data, self.cloudfree_data),
                  'MAE': MAE(self.pred_cloudfree_data.data, self.cloudfree_data),
                  }
        return scores

    def save_checkpoint(self, epoch):
        self.save_network(self.net_G, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)
    
    def load_checkpoint(self, epoch):
        checkpoint = torch.load(os.path.join(self.opts.save_model_dir, '%s_net_CR.pth' % (str(epoch))))
        self.net_G.load_state_dict(checkpoint['network'])
