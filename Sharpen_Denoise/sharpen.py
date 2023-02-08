import numpy as np
import cv2 
import sys
import argparse

def sharpen(image, kernel_size=(5, 5), sigma=1.0, amount=5, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255*np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def main():
    argList = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path")
    parser.add_argument("-o", "--out_path")
    parser.add_argument("-s", "--sf", type=int)
    args = parser.parse_args()

    inPath = "/content/drive/MyDrive/inputs/input.png"
    outPath = "/content/drive/MyDrive/outputs/output.png"

    if args.in_path:
        inPath = args.in_path
    if args.out_path:
        outPath = args.out_path

    print("PERFORMING SHARPENING ON SUPER RESOLVED IMAGE USING GAUSSIAN BLUR")
    
    img = cv2.imread(inPath)
    print(f"Input image shape={img.shape}")

    sharpened_img = sharpen(img, (80,80), 5)

    print(f"Output image shape={sharpened_img.shape}")

    cv2.imwrite(outPath, sharpened_img)

if __name__ == "__main__":
    main()
