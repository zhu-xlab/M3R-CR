This repository contains the codes associated with DSen2-CR in PyTorch.


If you use the codes for your research, please cite accordingly:

```
@article{meraner2020cloud,
  title={Cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion},
  author={Meraner, Andrea and Ebel, Patrick and Zhu, Xiao Xiang and Schmitt, Michael},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={166},
  pages={333--346},
  year={2020},
  publisher={Elsevier}
}
``` 

## Prerequisites & Installation

This code has been tested with CUDA 10.1 and Python 3.7.

```
conda create -n DSen2-CR python=3.7
pip install rasterio
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

## Get Started
You can download the pretrained model from [It will be will be released soon] and put it in './cpkg'.

Use the following command to test the neural network:
```
python test_CR.py
```

## Credits

This code is based on the codes available in the [EDSR-PyTorch
](https://github.com/sanghyun-son/EDSR-PyTorch) repo. I am grateful to the authors for making the original source code available.