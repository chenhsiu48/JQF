# JQF: JPEG Quantization Table Fusion

This repository contains the JPEG quantization table fusion code and pretrained weights for the paper [JQF: Optimal JPEG Quantization Table Fusion by Simulated Annealing on Texture Images and Predicting Textures](https://arxiv.org/abs/2008.05672). 

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

Simply execute the script ```predict.py``` with a list of images to encode, defaults to JPEG quality at Q=80. This program predicts the input image's texture distribution first, then fuses a per image optimized quantization table at targeted quality. 

```
$ python3 predict.py lighthouse.png
load ./data/JQF-Texture.pth to cuda
texture_id 49: 18.18%
texture_id 55: 11.69%
texture_id 73: 10.39%
texture_id 2: 10.39%
texture_id 59: 6.49%
...
...
fused table
  29   23   25   25   33   43   53   59 
  27   25   24   31   32   57   61   58 
  27   25   28   33   42   56   68   59 
  27   27   31   34   50   88   81   64 
  27   31   42   59   70  109  102   79 
  31   38   57   64   81  104  112   94 
  49   65   77   88  103  121  120  102 
  74   92   95   97  113  101  101   99 

scaled table at Q=80
  12    9   10   10   13   17   21   24 
  11   10   10   12   13   23   24   23 
  11   10   11   13   17   22   27   24 
  11   11   12   14   20   35   32   26 
  11   12   17   24   28   44   41   32 
  12   15   23   26   32   42   45   38 
  20   26   31   35   41   48   48   41 
  30   37   38   39   45   40   40   40 

save ./lighthouse-80-predict.jpg, 55962 bytes
```

You can override the default quality with argument ```-q 95```:  

```
$ python3 predict.py -q 95 r69076d90t.png
...
```

## Performance

### Quality

### Speed 

We used a workstation with Intel Core i7-9700K CPU and Nvidia
GeForce RTX 2080 Ti GPU for the benchmark. Time is reported in seconds. 

| Database          | # Images | GPU time | CPU time |
|-------------------|---------:|---------:|---------:|
| RAISE testing set |       50 |   0.4839 |  23.8342 |
| Kodak dataset     |       24 |   0.0438 |   2.6862 |
