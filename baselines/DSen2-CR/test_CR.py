import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import argparse
import numpy as np
import csv
from metrics import PSNR, SSIM, SAM, MAE, get_SAM_with_landcover, get_MAE_with_landcover

from dataloader import get_filelist, ValDataset

##########################################################
def test(CR_net, opts):

    test_filelist = get_filelist(opts.test_list_filepath)
    
    test_data = ValDataset(opts, test_filelist)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=opts.batch_sz, shuffle=False)

    iters = 0
    PSNR_4 = 0
    SSIM_4 = 0
    SAM_4 = 0
    MAE_4 = 0

    with torch.no_grad():
        for inputs in test_dataloader:

            cloudy_data = inputs['cloudy_data'].cuda()
            cloudfree_data = inputs['cloudfree_data'].cuda()
            SAR_data = inputs['SAR_data'].cuda()
            if opts.is_load_landcover:
                landcover_data = inputs['landcover_data'].cuda()
            file_name = inputs['file_name'][0]

            pred_cloudfree_data = CR_net(cloudy_data, SAR_data)
            
            results = [file_name]

            psnr_4 = PSNR(pred_cloudfree_data, cloudfree_data)

            ssim_4 = SSIM(pred_cloudfree_data, cloudfree_data)

            pixel_errors = MAE(pred_cloudfree_data, cloudfree_data, False)
            mae_4 = pixel_errors.mean()
            if opts.is_load_landcover:
                maes_4 = get_MAE_with_landcover(pixel_errors, landcover_data, opts.lc_level)

            angles = SAM(pred_cloudfree_data, cloudfree_data, False)
            sam_4 = angles.mean()
            if opts.is_load_landcover:
                sams_4 = get_SAM_with_landcover(angles, landcover_data, opts.lc_level)

            PSNR_4 += psnr_4
            SSIM_4 += ssim_4
            SAM_4 += sam_4
            MAE_4 += mae_4
            print(f'{iters}: PSNR:{psnr_4:.4f}, SSIM:{ssim_4:.4f}, SAM:{sam_4:.4f}, MAE:{mae_4:.4f}')

            iters += 1

    print('Testing done. ')
    print(f'PSNR:{PSNR_4/iters:.4f}, SSIM:{SSIM_4/iters:.4f}, SAM:{SAM_4/iters:.4f}, MAE:{MAE_4/iters:.4f}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')

    parser.add_argument('--n_resblocks', type=int, default=16)
    parser.add_argument('--n_feats', type=int, default=256)
    parser.add_argument('--res_scale', type=float, default=0.1)

    parser.add_argument('--input_data_folder', type=str, default='../../Planet-CR/test')
    parser.add_argument('--test_list_filepath', type=str, default='../../Planet-CR/one_test_sample.csv')
    parser.add_argument('--is_load_SAR', type=bool, default=True)
    parser.add_argument('--is_upsample_SAR', type=bool, default=True) # only useful when is_load_SAR = True
    parser.add_argument('--is_load_landcover', type=bool, default=False)
    parser.add_argument('--is_upsample_landcover', type=bool, default=False) # only useful when is_load_landcover = True
    parser.add_argument('--lc_level', type=str, default='2')  # only useful when is_load_landcover = True
    parser.add_argument('--is_load_cloudmask', type=bool, default=False)
    parser.add_argument('--load_size', type=int, default=300)
    parser.add_argument('--crop_size', type=int, default=300)

    opts = parser.parse_args()

    from EDSR import EDSR
    CR_net = EDSR(opts).cuda()
    CR_net = nn.DataParallel(CR_net)

    checkpoint = torch.load('./cpkg/DSen2-CR.pth')
    CR_net.load_state_dict(checkpoint['network'])

    CR_net.eval()

    test(CR_net, opts)

if __name__ == "__main__":
    main()
