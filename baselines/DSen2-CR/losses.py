import torch
import torch.nn as nn

class CARLLoss(nn.Module):
    def __init__(self):
        super(CARLLoss, self).__init__()
        self._weight = 1.0

    def forward(self, pred_cloudfree, cloudfree, cloudy, cloudmask):
        batch_size = pred_cloudfree.shape[0]
        loss = 0.
        for i in range(batch_size):
            _pred_cloudfree = pred_cloudfree[i, ...]
            _cloudfree = cloudfree[i, ...]
            _cloudy = cloudy[i, ...]
            _cloudmask = cloudmask[i, ...]
            _clearmask = torch.ones_like(_cloudmask) - _cloudmask
            loss += torch.mean(_clearmask*torch.abs(_pred_cloudfree-_cloudy)+_cloudmask*torch.abs(_pred_cloudfree-_cloudfree)) \
                    + self._weight*torch.mean(torch.abs(_pred_cloudfree-_cloudfree))
        return loss / batch_size
