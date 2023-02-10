This repository contains the codes associated with GLF-CR.


If you use the codes for your research, please cite accordingly:

```
@article{xu2022glf,
  title={GLF-CR: SAR-enhanced cloud removal with global--local fusion},
  author={Xu, Fang and Shi, Yilei and Ebel, Patrick and Yu, Lei and Xia, Gui-Song and Yang, Wen and Zhu, Xiao Xiang},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={192},
  pages={268--278},
  year={2022},
  publisher={Elsevier}
}
``` 

## Prerequisites & Installation

This code has been tested with CUDA 11.0 and Python 3.7.

```
conda create -n GLF-CR python=3.7
pip install rasterio
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm
```

## Get Started
You can download the pretrained model from [It will be released soon] and put it in './cpkg'.

Use the following command to test the neural network:
```
python test_CR.py
```

## Credits

This code is based on the codes available in the [GLF-CR
](https://github.com/xufangchn/GLF-CR) repo.
