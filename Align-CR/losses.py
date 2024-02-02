import torch
import torch.nn as nn

class ECCharbonnierLoss(nn.Module):
    def __init__(self):
        super(ECCharbonnierLoss, self).__init__()
        self._alpha = 0.45
        self._epsilon = 1e-3
        self.EC_weight = 5.
    
    def forward(self, pred_cloudfree, cloudfree, cloudmask):
        batch_size = pred_cloudfree.shape[0]
        loss = 0.
        for i in range(batch_size):
            _pred_cloudfree = pred_cloudfree[i, ...]
            _cloudfree = cloudfree[i, ...]
            _cloudmask = cloudmask[i, ...]
            _weight = torch.ones_like(_cloudmask) + self.EC_weight*_cloudmask
            loss += torch.mean(_weight*torch.pow(((_pred_cloudfree-_cloudfree) ** 2 + self._epsilon ** 2), self._alpha))
        return loss/batch_size