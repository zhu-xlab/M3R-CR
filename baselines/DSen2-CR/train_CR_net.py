import os
import sys
import torch
import argparse
import numpy as np

from dataloader import get_filelist, TrainDataset, ValDataset
from model_CR_net import ModelCRNet
from generic_train import Generic_Train
from model_base import print_options, seed_torch

##===================================================##
##********** Configure training settings ************##
##===================================================##
parser=argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')

parser.add_argument('--n_resblocks', type=int, default=16)
parser.add_argument('--n_feats', type=int, default=256)
parser.add_argument('--res_scale', type=float, default=0.1)

parser.add_argument('--input_data_folder', type=str, default='../../Planet-CR/train')
parser.add_argument('--train_list_filepath', type=str, default='../../Planet-CR/one_train_sample.csv')
parser.add_argument('--val_list_filepath', type=str, default='../../Planet-CR/one_train_sample.csv')
parser.add_argument('--is_load_SAR', type=bool, default=True)
parser.add_argument('--is_upsample_SAR', type=bool, default=True) # only useful when is_load_SAR = True
parser.add_argument('--is_load_landcover', type=bool, default=False)
parser.add_argument('--is_upsample_landcover', type=bool, default=False) # only useful when is_load_landcover = True
parser.add_argument('--lc_level', type=str, default='1')  # only useful when is_load_landcover = True
parser.add_argument('--is_load_cloudmask', type=bool, default=True)
parser.add_argument('--load_size', type=int, default=300)
parser.add_argument('--crop_size', type=int, default=160)

parser.add_argument('--optimizer', type=str, default='Adam', help = 'Adam')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of optimizer')
parser.add_argument('--lr_step', type=int, default=5, help='lr decay rate')
parser.add_argument('--lr_start_epoch_decay', type=int, default=10, help='epoch to start lr decay')
parser.add_argument('--max_epochs', type=int, default=30)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--val_freq', type=int, default=2)
parser.add_argument('--log_iter', type=int, default=10)
parser.add_argument('--save_model_dir', type=str, default='./checkpoints/', help='directory used to store trained networks')

parser.add_argument('--gpu_ids', type=str, default='0')

opts = parser.parse_args()
print_options(opts)

##===================================================##
##****************** choose gpu *********************##
##===================================================##
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids

##===================================================##
##*************** Create dataloader *****************##
##===================================================##
seed_torch()

train_filelist = get_filelist(opts.train_list_filepath)
val_filelist = get_filelist(opts.val_list_filepath)

train_data = TrainDataset(opts, train_filelist)
val_data = ValDataset(opts, val_filelist)
print("Train set: %d, Val set: %d" % (len(train_data), len(val_data)))

train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=opts.batch_sz,shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(dataset=val_data, batch_size=opts.batch_sz,shuffle=False, num_workers=4)

##===================================================##
##****************** Create model *******************##
##===================================================##
model=ModelCRNet(opts)

##===================================================##
##**************** Train the network ****************##
##===================================================##
Generic_Train(model, opts, train_dataloader, val_dataloader).train()





	