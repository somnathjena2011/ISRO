import torch

from basicsr_.models import create_model
from basicsr_.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr_.utils.options import parse
import numpy as np
import cv2
import matplotlib.pyplot as plt

import argparse

def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def display(img1, img2):
    fig = plt.figure(figsize=(25, 10))
    ax1 = fig.add_subplot(1, 2, 1) 
    plt.title('Input image', fontsize=16)
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.title('NAFNet output', fontsize=16)
    ax2.axis('off')
    ax1.imshow(img1)
    ax2.imshow(img2)

def single_image_inference(model, img, save_path, tile=None):
    if tile is not None:
        img_ = torch.unsqueeze(img, 0)
        b, c, h, w = img_.size()
        tile = min(tile, h, w)
        window_size = 8
        tile_overlap = 32
        assert tile % window_size == 0, "tile size should be a multiple of window_size"

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h, w).type_as(img)
        W = torch.zeros_like(E)

        print(f"Input image shape={img.shape}")

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img[..., h_idx:h_idx+tile, w_idx:w_idx+tile]


                model.feed_data(data={'lq': in_patch.unsqueeze(dim=0)})

                if model.opt['val'].get('grids', False):
                    model.grids()

                model.test()

                if model.opt['val'].get('grids', False):
                    model.grids_inverse()

                visuals = model.get_current_visuals()
                #print(f"shaping={visuals['result'].size()}")
                #out_patch = tensor2img([visuals['result']])
                out_patch = visuals["result"]

                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
        sr_img = tensor2img(E.div_(W))
        print(f"Output shape={sr_img.shape}")
        imwrite(sr_img, save_path)
    
    else:
        model.feed_data(data={'lq': img.unsqueeze(dim=0)})

        if model.opt['val'].get('grids', False):
            model.grids()

        model.test()

        if model.opt['val'].get('grids', False):
            model.grids_inverse()

        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        print(f"de shape={sr_img.shape} path={save_path}")
        print(f"Output shape={sr_img.shape}")
        imwrite(sr_img, save_path)

def main():
    opt_path = 'options/test/SIDD/NAFNet-width64.yml'
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    NAFNet = create_model(opt)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path")
    parser.add_argument("-o", "--out_path")
    parser.add_argument("-t", "--tile", type=int, default=None)
    args = parser.parse_args()

    inPath = "/content/drive/MyDrive/inputs/input.png"
    outPath = "/content/drive/MyDrive/outputs/output.png"
    tile = None

    if args.in_path:
        inPath = args.in_path
    if args.out_path:
        outPath = args.out_path
    if args.tile is not None:
        tile = args.tile
    
    print("PERFORMING DENOISING USING NAFNET")

    img_input = imread(inPath)
    inp = img2tensor(img_input)
    # test the image tile by tile
    
    single_image_inference(NAFNet, inp, outPath, tile)

if __name__ == "__main__":
    main()
