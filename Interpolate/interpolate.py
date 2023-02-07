import cv2
import numpy as np
import argparse
import sys

argList = sys.argv[1:]

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--in_path")
parser.add_argument("-o", "--out_path")
parser.add_argument("-s", "--sf", type=int)
args = parser.parse_args()
sf = 4
if args.sf:
  sf = args.sf

inPath = "/content/drive/MyDrive/Super_Resolution_code/KAIR-master/superresolution/super_resolution_result/test1_msrresnet_x4_psnr.png"
outPath = "/content/drive/MyDrive/outputs/output.png"

if args.in_path:
  inPath = args.in_path
if args.out_path:
  outPath = args.out_path

print("inPath")
print(inPath)

img = cv2.imread(inPath)
print(f"shape={img.shape}")

bicubic_img = cv2.resize(img,None, fx = sf, fy = sf, interpolation = cv2.INTER_CUBIC)

cv2.imwrite(outPath, bicubic_img)
