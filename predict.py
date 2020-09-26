#!/usr/bin/env python3

from argparse import ArgumentParser
import torch
from torch import nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import utils

BK_SIZE = 64
TEXTURE_CATS = 100
BATCH_SIZE = 256

texture_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

std_chroma = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99
]


class TextureNet(nn.Module):
    def __init__(self, input_size, cat_num):
        super(TextureNet, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        for param in self.vgg16.features.parameters():
            param.requires_grad = False
        for param in self.vgg16.avgpool.parameters():
            param.requires_grad = False
        self.vgg16.classifier[0] = nn.Linear((input_size // 32) * (input_size // 32) * 512, 4096)
        self.vgg16.classifier[6] = nn.Linear(4096, cat_num)

    def forward(self, x):
        x = self.vgg16.features(x)
        h = x.view(x.shape[0], -1)
        return self.vgg16.classifier(h)

    @staticmethod
    def loss_fn(y_pred, y):
        return nn.functional.cross_entropy(y_pred, y.squeeze(1))


def non_overlap_crop(im, patch_size=BK_SIZE, stride=BK_SIZE):
    w, h = im.size
    patches = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = texture_transform(im.crop((j, i, j + patch_size, i + patch_size)))
            patches = patches + (patch,)
    return patches


def exec_encode(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'load {args.model} to {device}')
    model = TextureNet(BK_SIZE, TEXTURE_CATS).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    qtables = utils.load_qtables(args)

    model.eval()
    with torch.no_grad():
        for img_name in args.images:
            pred_name = utils.make_filepath(img_name, ext_name='jpg', tag=f'{args.quality}-predict')

            im = Image.open(img_name)
            all_patches = non_overlap_crop(im)
            num_blocks = len(all_patches)

            bk_hist = np.zeros(TEXTURE_CATS, dtype=int)
            for p in range(0, num_blocks, BATCH_SIZE):
                batch = all_patches[p:p + BATCH_SIZE]
                y_pred = model(torch.stack(batch).to(device))
                y_pred = nn.functional.softmax(y_pred, dim=1)
                y_pred = y_pred.argmax(dim=1).view(y_pred.size(0), -1)
                for i in range(len(y_pred)):
                    bk_hist[y_pred[i][0]] += 1

            for th in sorted(zip(bk_hist, range(len(bk_hist))), reverse=True):
                if th[0] > 0:
                    print(f'texture_id {th[1]:d}: {100 * th[0] / num_blocks:.2f}%')

            fuse_luma, fuse_chroma = np.zeros(64), np.zeros(64)
            for i in range(len(bk_hist)):
                q_luma = qtables[str(i)]
                fuse_luma = fuse_luma + np.multiply(q_luma, float(bk_hist[i]) / num_blocks)
            fuse_luma = list(map(int, np.round(fuse_luma)))
            print('fused table')
            utils.print_qtable(fuse_luma)
            fuse_luma_scaled = utils.scale_qtable(fuse_luma, args.quality)
            print(f'scaled table at Q={args.quality}')
            utils.print_qtable(fuse_luma_scaled)

            std_chroma_scaled = utils.scale_qtable(std_chroma, args.quality)

            im.save(pred_name, quality=utils.quality_to_scale(50), qtables=[fuse_luma_scaled, std_chroma_scaled])
            pred_size = os.path.getsize(pred_name)
            print(f'save {pred_name}, {pred_size} bytes')


if __name__ == '__main__':
    parser = ArgumentParser(description='')
    parser.add_argument('images', nargs='+', help='images')
    parser.add_argument('--model', default='./data/JQF-Texture.pth', type=str, help='texture model')
    parser.add_argument('--qtable', default='./data/JQF-qtables.txt', type=str, help='optimized qtables')
    parser.add_argument('--quality', '-q', default=80, type=int, help='JPEG encoding quality metric')
    args = parser.parse_args()
    exec_encode(args)
