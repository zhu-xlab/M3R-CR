import os
import csv
import rasterio
import numpy as np
import random
import argparse

import torch
from torch.utils.data import Dataset

'''
'''
def get_opt_image(path):

    src = rasterio.open(path, 'r', driver='GTiff')
    image = src.read()
    src.close()
    image[np.isnan(image)] = np.nanmean(image)  # fill holes and artifacts

    return image.astype('float32')

'''
'''
def get_sar_image(path, load_size, is_upsample_SAR):

    src = rasterio.open(path, 'r', driver='GTiff')
    if is_upsample_SAR:
        image = src.read(out_shape = (src.count, load_size, load_size), resampling = rasterio.enums.Resampling.nearest)
    else:
        image = src.read()
    src.close()
    image[np.isnan(image)] = np.nanmean(image)  # fill holes and artifacts

    return image.astype('float32')
 
'''
'''   
def get_landcover_image(path, load_size, is_upsample_landcover):
    
    src = rasterio.open(path, 'r', driver='GTiff')
    if is_upsample_landcover:
        image = src.read(out_shape = (src.count, load_size, load_size), resampling = rasterio.enums.Resampling.nearest)
        image = image[0, :, :]
    else:
        image = src.read(1)
    src.close()
    image[np.isnan(image)] = np.nanmean(image)  # fill holes and artifacts

    return image.astype('float32')

'''
'''
def get_normalized_data(data_image, data_type):

    clip_min = [[-25.0, -32.5], [0, 0, 0, 0], [0, 0, 0, 0]]
    clip_max = [[0, 0], [10000, 10000, 10000, 10000], [10000, 10000, 10000, 10000]]

    max_val = 1
    scale = 10000

    # SAR
    if data_type == 1:
        for channel in range(len(data_image)):
            data_image[channel] = np.clip(data_image[channel], clip_min[data_type - 1][channel], clip_max[data_type - 1][channel])
            data_image[channel] -= clip_min[data_type - 1][channel]
            data_image[channel] = max_val * (data_image[channel] / (clip_max[data_type - 1][channel] - clip_min[data_type - 1][channel]))
    # OPT
    elif data_type == 2 or data_type == 3:
        for channel in range(len(data_image)):
            data_image[channel] = np.clip(data_image[channel], clip_min[data_type - 1][channel], clip_max[data_type - 1][channel])
        data_image /= scale

    return data_image

'''
'''
def lc_category_map(data_image, lc_level):
    '''
        Trees = [0, 100, 0]                          # 10	006400	Trees
        Shrubland = [255, 187, 34]                   # 20	ffbb22	Shrubland
        Grassland = [255, 255, 76]                   # 30	ffff4c	Grassland
        Cropland = [240, 150, 255]                   # 40	f096ff	Cropland
        Built_up = [250, 0, 0]                       # 50	fa0000	Built-up
        Barren_sparse_vegetation = [180, 180, 180]   # 60	b4b4b4	Barren / sparse vegetation
        Snow_and_ice = [240, 240, 240]               # 70	f0f0f0	Snow and ice
        Open_water = [0, 100, 200]                   # 80	0064c8	Open water
        Herbaceous_wetland = [0, 150, 160]           # 90	0096a0	Herbaceous wetland
        Mangroves = [0, 207, 117]                    # 95	00cf75	Mangroves
        Moss_and_lichen = [250, 230, 160]            # 100	fae6a0	Moss and lichen
    '''
    if lc_level == '1':
        data_image[data_image == 10] = 0 # Trees --> Forest land
        data_image[data_image == 20] = 1 # Shrubland --> Rangeland
        data_image[data_image == 30] = 1 # Grassland --> Rangeland
        data_image[data_image == 40] = 2 # Cropland --> Agriculture land
        data_image[data_image == 50] = 3 # Built-up --> Urban land
        data_image[data_image == 60] = 4 # Barren / sparse vegetation --> Barren land
        data_image[data_image == 70] = 6 # Snow and ice --> Unknown
        data_image[data_image == 80] = 5 # Open water --> Water
        data_image[data_image == 90] = 5 # Herbaceous wetland --> Water
        data_image[data_image == 95] = 6 # Mangroves --> Unknown
        data_image[data_image == 100] = 6 # Moss and lichen --> Unknown
    elif lc_level == '2':
        data_image[data_image == 10] = 0
        data_image[data_image == 20] = 1
        data_image[data_image == 30] = 2
        data_image[data_image == 40] = 3
        data_image[data_image == 50] = 4
        data_image[data_image == 60] = 5
        data_image[data_image == 70] = 6
        data_image[data_image == 80] = 7
        data_image[data_image == 90] = 8
        data_image[data_image == 95] = 9
        data_image[data_image == 100] = 10
    return data_image

'''
'''
class TrainDataset(Dataset):

    def __init__(self, opts, filelist):

        self.input_data_folder = opts.input_data_folder
        self.is_load_SAR = opts.is_load_SAR
        self.is_upsample_SAR = opts.is_upsample_SAR
        self.is_load_landcover = opts.is_load_landcover
        self.is_upsample_landcover = opts.is_upsample_landcover
        self.lc_level = opts.lc_level
        self.is_load_cloudmask = opts.is_load_cloudmask
        self.load_size = opts.load_size
        self.crop_size = opts.crop_size

        self.filelist = filelist
        self.n_images = len(self.filelist)

    def __getitem__(self, index):

        [planet_cloudfree_path, planet_cloudy_path, S1_path, landcover_path, cloudmask_path] = self.filelist[index]

        planet_cloudfree_path = os.path.join(self.input_data_folder, planet_cloudfree_path)
        planet_cloudy_path = os.path.join(self.input_data_folder, planet_cloudy_path)
        if self.is_load_SAR:
            s1_path = os.path.join(self.input_data_folder, S1_path)
        if self.is_load_landcover:
            lc_path = os.path.join(self.input_data_folder, landcover_path)
        if self.is_load_cloudmask:
            cm_path = os.path.join(self.input_data_folder, cloudmask_path)

        planet_cloudfree_data = get_opt_image(planet_cloudfree_path)
        planet_cloudy_data = get_opt_image(planet_cloudy_path)
        if self.is_load_SAR:
            s1_data = get_sar_image(s1_path, self.load_size, self.is_upsample_SAR)
        if self.is_load_landcover:
            lc_data = get_landcover_image(lc_path, self.load_size, self.is_upsample_landcover)
        if self.is_load_cloudmask:
            cm_data = np.load(cm_path, encoding='bytes', allow_pickle=True).astype('float32')

        planet_cloudfree_data = get_normalized_data(planet_cloudfree_data, data_type=2)
        planet_cloudy_data = get_normalized_data(planet_cloudy_data, data_type=3)
        if self.is_load_SAR:
            s1_data = get_normalized_data(s1_data, data_type=1)
        if self.is_load_landcover:
            lc_data = lc_category_map(lc_data, self.lc_level)

        planet_cloudfree_data = torch.from_numpy(planet_cloudfree_data)
        planet_cloudy_data = torch.from_numpy(planet_cloudy_data)
        if self.is_load_SAR:
            s1_data = torch.from_numpy(s1_data)
        if self.is_load_landcover:
            lc_data = torch.from_numpy(lc_data)
        if self.is_load_cloudmask:
            cm_data = torch.from_numpy(cm_data)

        if self.load_size - self.crop_size > 0:
            if (self.is_load_SAR and not self.is_upsample_SAR) or (self.is_load_landcover and not self.is_upsample_landcover):
                y = random.randint(0, np.maximum(0, (self.load_size - self.crop_size)//10))
                x = random.randint(0, np.maximum(0, (self.load_size - self.crop_size)//10))
                y_3m, x_3m, y_10m, x_10m = y*10, x*10, y*3, x*3
                if self.is_load_SAR:
                    if self.is_upsample_SAR:
                        s1_data = s1_data[..., y_3m:y_3m+self.crop_size, x_3m:x_3m+self.crop_size]
                    else:
                        s1_data = s1_data[..., y_10m:y_10m+int(self.crop_size*3/10), x_10m:x_10m+int(self.crop_size*3/10)]
                if self.is_load_landcover:
                    if self.is_upsample_landcover:
                        lc_data = lc_data[..., y_3m:y_3m+self.crop_size, x_3m:x_3m+self.crop_size]
                    else:
                        lc_data = lc_data[..., y_10m:y_10m+int(self.crop_size*3/10), x_10m:x_10m+int(self.crop_size*3/10)]
            else:
                y = random.randint(0, np.maximum(0, self.load_size - self.crop_size))
                x = random.randint(0, np.maximum(0, self.load_size - self.crop_size))
                y_3m, x_3m = y, x
                if self.is_load_SAR:
                    s1_data = s1_data[..., y_3m:y_3m+self.crop_size, x_3m:x_3m+self.crop_size]
                if self.is_load_landcover:
                    lc_data = lc_data[..., y_3m:y_3m+self.crop_size, x_3m:x_3m+self.crop_size]
            
            planet_cloudfree_data = planet_cloudfree_data[..., y_3m:y_3m+self.crop_size, x_3m:x_3m+self.crop_size]
            planet_cloudy_data = planet_cloudy_data[..., y_3m:y_3m+self.crop_size, x_3m:x_3m+self.crop_size]
            if self.is_load_cloudmask:
                cm_data = cm_data[y_3m:y_3m+self.crop_size, x_3m:x_3m+self.crop_size]

        results = {'cloudy_data': planet_cloudy_data,
                   'cloudfree_data': planet_cloudfree_data,
                   'file_name': os.path.basename(planet_cloudfree_path)}
        if self.is_load_SAR:
            results['SAR_data'] = s1_data
        if self.is_load_landcover:
            results['landcover_data'] = lc_data
        if self.is_load_cloudmask:
            results['cloudmask_data'] = cm_data

        return results

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.n_images

'''
'''
class ValDataset(Dataset):

    def __init__(self, opts, filelist):

        self.input_data_folder = opts.input_data_folder
        self.is_load_SAR = opts.is_load_SAR
        self.is_upsample_SAR = opts.is_upsample_SAR
        self.is_load_landcover = opts.is_load_landcover
        self.is_upsample_landcover = opts.is_upsample_landcover
        self.lc_level = opts.lc_level
        self.is_load_cloudmask = opts.is_load_cloudmask
        self.load_size = opts.load_size
        self.crop_size = opts.crop_size

        self.filelist = filelist
        self.n_images = len(self.filelist)

    def __getitem__(self, index):

        [planet_cloudfree_path, planet_cloudy_path, S1_path, landcover_path, cloudmask_path] = self.filelist[index]

        planet_cloudfree_path = os.path.join(self.input_data_folder, planet_cloudfree_path)
        planet_cloudy_path = os.path.join(self.input_data_folder, planet_cloudy_path)
        if self.is_load_SAR:
            s1_path = os.path.join(self.input_data_folder, S1_path)
        if self.is_load_landcover:
            lc_path = os.path.join(self.input_data_folder, landcover_path)
        if self.is_load_cloudmask:
            cm_path = os.path.join(self.input_data_folder, cloudmask_path)

        planet_cloudfree_data = get_opt_image(planet_cloudfree_path)
        planet_cloudy_data = get_opt_image(planet_cloudy_path)
        if self.is_load_SAR:
            s1_data = get_sar_image(s1_path, self.load_size, self.is_upsample_SAR)
        if self.is_load_landcover:
            lc_data = get_landcover_image(lc_path, self.load_size, self.is_upsample_landcover)
        if self.is_load_cloudmask:
            cm_data = np.load(cm_path, encoding='bytes', allow_pickle=True).astype('float32')

        planet_cloudfree_data = get_normalized_data(planet_cloudfree_data, data_type=2)
        planet_cloudy_data = get_normalized_data(planet_cloudy_data, data_type=3)
        if self.is_load_SAR:
            s1_data = get_normalized_data(s1_data, data_type=1)
        if self.is_load_landcover:
            lc_data = lc_category_map(lc_data, self.lc_level)

        planet_cloudfree_data = torch.from_numpy(planet_cloudfree_data)
        planet_cloudy_data = torch.from_numpy(planet_cloudy_data)
        if self.is_load_SAR:
            s1_data = torch.from_numpy(s1_data)
        if self.is_load_landcover:
            lc_data = torch.from_numpy(lc_data)
        if self.is_load_cloudmask:
            cm_data = torch.from_numpy(cm_data)

        if self.load_size - self.crop_size > 0:
            if (self.is_load_SAR and not self.is_upsample_SAR) or (self.is_load_landcover and not self.is_upsample_landcover):
                y = np.maximum(0, (self.load_size - self.crop_size)//10)//2
                x = np.maximum(0, (self.load_size - self.crop_size)//10)//2
                y_3m, x_3m, y_10m, x_10m = y*10, x*10, y*3, x*3
                if self.is_load_SAR:
                    if self.is_upsample_SAR:
                        s1_data = s1_data[..., y_3m:y_3m+self.crop_size, x_3m:x_3m+self.crop_size]
                    else:
                        s1_data = s1_data[..., y_10m:y_10m+int(self.crop_size*3/10), x_10m:x_10m+int(self.crop_size*3/10)]
                if self.is_load_landcover:
                    if self.is_upsample_landcover:
                        lc_data = lc_data[..., y_3m:y_3m+self.crop_size, x_3m:x_3m+self.crop_size]
                    else:
                        lc_data = lc_data[..., y_10m:y_10m+int(self.crop_size*3/10), x_10m:x_10m+int(self.crop_size*3/10)]
            else:
                y = np.maximum(0, self.load_size - self.crop_size)//2
                x = np.maximum(0, self.load_size - self.crop_size)//2
                y_3m, x_3m = y, x
                if self.is_load_SAR:
                    s1_data = s1_data[..., y_3m:y_3m+self.crop_size, x_3m:x_3m+self.crop_size]
                if self.is_load_landcover:
                    lc_data = lc_data[..., y_3m:y_3m+self.crop_size, x_3m:x_3m+self.crop_size]
            
            planet_cloudfree_data = planet_cloudfree_data[..., y_3m:y_3m+self.crop_size, x_3m:x_3m+self.crop_size]
            planet_cloudy_data = planet_cloudy_data[..., y_3m:y_3m+self.crop_size, x_3m:x_3m+self.crop_size]
            if self.is_load_cloudmask:
                cm_data = cm_data[y_3m:y_3m+self.crop_size, x_3m:x_3m+self.crop_size]

        results = {'cloudy_data': planet_cloudy_data,
                   'cloudfree_data': planet_cloudfree_data,
                   'file_name': os.path.basename(planet_cloudfree_path)}
        if self.is_load_SAR:
            results['SAR_data'] = s1_data
        if self.is_load_landcover:
            results['landcover_data'] = lc_data
        if self.is_load_cloudmask:
            results['cloudmask_data'] = cm_data
            
        return results

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.n_images

'''
'''
def get_filelist(listpath):

    filelist = []
    list_csv_file = open(listpath, "r")
    list_reader = csv.reader(list_csv_file)
    for item in list_reader:
        filelist.append(item)
    list_csv_file.close()

    return filelist

if __name__ == "__main__":
    ##===================================================##
    parser=argparse.ArgumentParser()
    parser.add_argument('--load_size', type=int, default=300)
    parser.add_argument('--crop_size', type=int, default=160)
    parser.add_argument('--input_data_folder', type=str, default='../../M3M-CR/train')

    parser.add_argument('--is_load_SAR', type=bool, default=True)
    parser.add_argument('--is_upsample_SAR', type=bool, default=True) # only useful when is_load_SAR = True

    parser.add_argument('--is_load_landcover', type=bool, default=True)
    parser.add_argument('--is_upsample_landcover', type=bool, default=True) # only useful when is_load_landcover = True
    parser.add_argument('--lc_level', type=str, default='1')  # only useful when is_load_landcover = True

    parser.add_argument('--is_load_cloudmask', type=bool, default=True)

    parser.add_argument('--data_list_filepath', type=str, default='../../M3M-CR/one_train_sample.csv')
    
    opts = parser.parse_args() 

    ##===================================================##
    train_filelist = get_filelist(opts.data_list_filepath)
    data = TrainDataset(opts, train_filelist)
    dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=1,shuffle=True)

    ##===================================================##
    _iter = 0
    for results in dataloader:
        cloudy_data = results['cloudy_data']
        cloudfree_data = results['cloudfree_data']
        file_name = results['file_name'][0]
        print(file_name)
        print('cloudy_data:', cloudy_data.shape)
        print('cloudfree_data', cloudfree_data.shape)

        if opts.is_load_SAR:
            s1_data = results['SAR_data']
            print('SAR_data:', s1_data.shape)
        if opts.is_load_landcover:
            lc_data = results['landcover_data']
            print('landcover_data:', lc_data.shape)
        if opts.is_load_cloudmask:
            cm_data = results['cloudmask_data']
            print('cloudmask_data:', cm_data.shape)
        
        _iter += 1