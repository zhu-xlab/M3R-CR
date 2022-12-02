import common
import os
import argparse

import torch.nn as nn
import torch

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        # scale = 2
        # act = nn.ReLU(True)
        # url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        # if url_name in url:
        #     self.url = url[url_name]
        # else:
        #     self.url = None
        # self.sub_mean = common.MeanShift(args.rgb_range)
        # self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(6, n_feats, kernel_size),
            nn.ReLU(True)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=nn.ReLU(True), res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, 4, kernel_size))

        # # define tail module
        # m_tail = [
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     conv(n_feats, args.n_colors, kernel_size)
        # ]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        # self.tail = nn.Sequential(*m_tail)

    def forward(self, opt, sar):
        # x = self.sub_mean(x)
        x = self.head(torch.cat((opt, sar),1))

        res = self.body(x)
        pred = opt + res

        # x = self.tail(res)
        # x = self.add_mean(x)

        return pred 

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    parser=argparse.ArgumentParser()
    parser.add_argument('--n_resblocks', type=int, default=16)
    parser.add_argument('--n_feats', type=int, default=256)
    parser.add_argument('--res_scale', type=float, default=0.1)
    parser.add_argument('--crop_size', type=int, default=160)
    args = parser.parse_args()

    model = EDSR(args).cuda()

    planet_cloudy = torch.rand(1, 4, 160, 160).cuda()
    s1_sar = torch.rand(1, 2, 160, 160).cuda()

    pred_planet_cloudfree = model(planet_cloudy, s1_sar)

    print(pred_planet_cloudfree.shape)