# JQF: JPEG Quantization Table Fusion

This repository contains the JPEG quantization table fusion code and pretrained model for the paper [JQF: Optimal JPEG Quantization Table Fusion by Simulated Annealing on Texture Images and Predicting Textures](https://arxiv.org/abs/2008.05672).

> This work tries to solve the dilemma of balancing between computational cost and image specific optimality by introducing a new concept of texture mosaic images. Instead of optimizing a single image or a collection of representative images, the simulated annealing technique is applied to texture mosaic images to search for an optimal quantization table for each texture category. We use pre-trained VGG-16 CNN model to learn those texture features and predict the new image's texture distribution, then fuse optimal texture tables to come out with an image specific optimal quantization table. On the Kodak dataset with the quality setting Q=95, our experiment shows a size reduction of 23.5% over the JPEG standard table with a slightly 0.35% FSIM decrease, which is visually unperceivable. The proposed JQF method achieves per image optimality for JPEG encoding with less than one second additional timing cost. The online demo is available at https://matthorn.s3.amazonaws.com/JQF/qtbl_vis.html.

## Citation

If you use the model in your research, please cite:
```
@article{huang2020jqf,
  title={JQF: Optimal JPEG Quantization Table Fusion by Simulated Annealing on Texture Images and Predicting Textures},
  author={Huang, Chen-Hsiu and Wu, Ja-Ling},
  journal={arXiv e-prints},
  pages={arXiv--2008},
  year={2020}
}
```

## Requirements

```
$ pip install -r requirements.txt
```

## Usage

First download the pretrained weights with:

```
$ wget -P data/ https://matthorn.s3-ap-northeast-1.amazonaws.com/JQF/JQF-Texture.pth
```

Simply execute the script ```predict.py``` with a list of images to encode, defaults to JPEG quality at Q=95. This program predicts the input image's texture distribution first, then fuses a per image optimized quantization table at targeted quality.

```
$ python3 predict.py image/lighthouse.png
load ./data/JQF-Texture.pth to cuda
texture_id 49: 18.18%
texture_id 55: 11.69%
texture_id 73: 10.39%
texture_id 2: 10.39%
texture_id 59: 6.49%
...
...
fused table
  59   58   46   62   57   61   66   75 
  48   58   50   50   60   84   75   72 
  50   50   52   63   61   81   81   79 
  50   61   59   63   77   92   87   80 
  47   54   63   76   87  109  111   89 
  62   66   76   77   90  107  118   93 
  62   82   90   99  111  124  121  111 
  86  100  103  100  114  104  110  108 

scaled table at Q=95
   6    6    5    6    6    6    7    8 
   5    6    5    5    6    8    8    7 
   5    5    5    6    6    8    8    8 
   5    6    6    6    8    9    9    8 
   5    5    6    8    9   11   11    9 
   6    7    8    8    9   11   12    9 
   6    8    9   10   11   12   12   11 
   9   10   10   10   11   10   11   11 

save ./lighthouse-95-jqf.jpg, 107502 bytes
```

There are two sets of texture quantization tables optimized at Q=50 and Q=95, named ```JQF-qtables-50.txt``` and ```JQF-qtables-95.txt``` under folder ```data/```. As shown in the paper, direct scale original annealed Q to different Q at encoding time is not optimal and will introduce visible artifact at low bitrate. Therefore we choose Q=50 as a compromise for a wide range of application scenarios and report the rate-distortion curve of ```lighthouse.png``` in the paper. However, since it is common to encode JPEG as a high-quality image, we set our default to use Q=95 optimized tables for its best bitrate saving. 

To override the default texture quantization table, use the argument ```--qtable```:

```
$ python3 predict.py --qtable data/JQF-qtables-50.txt image/r69076d90t.png
...
```

## Performance

### Rate-distortion Curves

#### lighthouse

![lighthouse](https://github.com/chenhsiu48/JQF/raw/master/RD/kodim19-rd.png)

#### bikes

![bikes](https://github.com/chenhsiu48/JQF/raw/master/RD/kodim05-rd.png)

#### parrots
![parrots](https://github.com/chenhsiu48/JQF/raw/master/RD/kodim23-rd.png)

#### caps
![caps](https://github.com/chenhsiu48/JQF/raw/master/RD/kodim03-rd.png)

#### womanhat
![womanhat](https://github.com/chenhsiu48/JQF/raw/master/RD/kodim04-rd.png)

### Prediction Speed

We used a workstation with Intel Core i7-9700K CPU and Nvidia
GeForce RTX 2080 Ti GPU for the benchmark. Time is reported in seconds.

| Database          | # Images | GPU time | CPU time |
|-------------------|---------:|---------:|---------:|
| RAISE testing set |       50 |   0.4839 |  23.8342 |
| Kodak dataset     |       24 |   0.0438 |   2.6862 |
