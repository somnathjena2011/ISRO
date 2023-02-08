# ISRO

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#installations">Installations</a></li>
    <li>
        <a href="#usage">Usage</a>
        <ul>
            <li><a href="#arguments-format">Arguments format</a></li>
        </ul>
    </li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This project demonstrates the implementation of Deep Learning models for Image Super-Resolution of Lunar Surface data captured from 2 cameras from the Chandrayan-2 Mission of ISRO: namely TMC-2 and OHRC. 
Terrain Mapping Camera-2 (TMC-2) Chadrayaan-2 Orbiter maps the lunar
surface in the panchromatic spectral band (0.5-0.8 microns) with a spatial resolution of 5 meter. The Orbiter Higher Resolution Camera (OHRC) onboard Chadrayaan-2 Orbiter is an imaging payloads which provides high resolution (~ 30 cm) images of lunar surface. The Deep Learning model is trained to perform image super-resolution (SR) of the low resolution images to obtain high resolution images (~16x).

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- PROJECT STRUCTURE -->

## Project Structure

```
.
├── ESRGAN
│   ├── assets
│   ├── cog_predict.py
│   ├── cog.yaml
│   ├── experiments
│   ├── inference_realesrgan.py
│   ├── inputs
│   ├── main_test_realesrgan.py
│   ├── MANIFEST.in
│   ├── options
│   ├── README.md
│   ├── realesrgan
│   ├── realesrgan.egg-info
│   ├── requirements.txt
│   ├── results
│   ├── scripts
│   ├── setup.cfg
│   ├── setup.py
│   ├── tb_logger
│   ├── tests
│   ├── VERSION
│   └── weights
├── HAT
│   ├── cog.yaml
│   ├── datasets
│   ├── experiments
│   ├── figures
│   ├── hat
│   ├── hat.egg-info
│   ├── options
│   ├── predict.py
│   ├── README.md
│   ├── requirements.txt
│   ├── results
│   ├── setup.cfg
│   ├── setup.py
│   └── VERSION
├── Interpolate
│   └── interpolate.py
├── LightEnhancement
│   ├── ckpt
│   ├── demo
│   ├── evaluate.py
│   ├── figure
│   ├── network
│   ├── README.md
│   ├── test.py
│   └── utils.py
├── MSR_SWINIR
│   ├── data
│   ├── docs
│   ├── figs
│   ├── kernels
│   ├── main_test_msrresnet.py
│   ├── main_test_swinir.py
│   ├── main_train_gan.py
│   ├── main_train_psnr.py
│   ├── matlab
│   ├── models
│   ├── options
│   ├── pretrained_weights
│   ├── README.md
│   ├── requirement.txt
│   ├── results
│   ├── retinaface
│   ├── scripts
│   ├── superresolution
│   └── utils
├── pipeline.py
├── README.md
├── requirements.txt
└── Sharpen_Denoise
    ├── NAFNet
    └── sharpen.py
```
<p align="right">(<a href="#top">back to top</a>)</p>


<!-- INSTALLATIONS -->

### Installations

```bash
# Clone the repository
git clone https://github.com/somnathjena2011/ISRO.git

# Change directory to repo directory
cd ISRO

# Download model weights from the following link
# https://drive.google.com/drive/folders/1wmWoJ2gYrbt6Fqkyr3x8oS-cvFEfCic1?usp=sharing
# Put the appropriate models in the appropriate weights folders

# Download requirements for SwinIR
cd MSR_SWINIR
pip install -r requirement.txt
cd ..

# Download and setup requirements for RealESRGAN
cd ESRGAN
pip install basicsr
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop
cd ..

# Download and setup requirements for HAT
cd HAT
pip install -r requirements.txt
python setup.py develop
cd ..

```

<!-- USAGE -->

## Usage

To run the super resolution code, you need to run the pipeline.py file
passing appropriate arguments. So the present working directory should be the
repository

```bash

python pipeline.py -i <input_file_path> -o <output_file_path> sr --sr_model <model_name> --scale <scale_factor> --tile <tile_size> int --scale <scale_factor> shp

```

### Arguments format

| **option**        | **alternate option** | **description**                                                       | **default**                |
|-------------------|----------------------|-----------------------------------------------------------------------|----------------------------|
| --input           | -i                   | input file path for single file input                                 | inputs/input.png           |
| --output          | -o                   | output file path                                                      | outputs                    |
| --input_folder    | -if                  | input folder consisting of only images                                | None                       |
| --output_folder   | -of                  | output folder to store output images                                  | None                       |
| --compress_output | -co                  | flag to indicate whether to save intermediate generated images or not | True                       |
| sr                |                      |                                                                       |                            |
| --sr_model        | -sm                  | name of SR model                                                      | realesrgan                 |
| --sr_path         | -sp                  | path to SR code                                                       | ESRGAN                     |
| --tile            | -t                   | tile size to avoid CUDA error                                         | None                       |
| --model_path      | -mp                  | path of weights                                                       | None                       |
| --scale           | s                    | scale factor                                                          | 4                          |
| int               |                      |                                                                       |                            |
| --int_model       | -im                  | name of interpolation method                                          | bicubic                    |
| --int_path        | -ip                  | path to interpolation code                                            | Interpolate/interpolate.py |
| --scale           | -s                   | scale factor                                                          | 4                          |
| enh               |                      |                                                                       |                            |
| --enh_path        | -ep                  | path containing light enhancement code                                | LightEnhancement           |
| --enh_model       | -em                  | light enhancement model                                               | URetinex                   |
| shp               |                      |                                                                       |                            |
| den               |                      |                                                                       |                            |
| --tile            | -t                   | tile size to avoid cuda error                                         | None                       |

<p align="right">(<a href="#top">back to top</a>)</p>

## Model Description

### Lunar Turing-GAN (T-GAN)


![Alt text](images/Lunar_T-Gan.png?raw=true "Figure describes the architecutre of our proposed Lunar Turing-GAN (T-GAN)")

#### Input to the model:

* Original HR Image
* Low Resolution image (Downsized from Original)
* Depth map of Original Image

We modify the conventional discriminator of conventional GANs with a novel turing loss that ensures the model places a special emphasis on the region of interest: in our case the craters and the hills. More specifically, as shown in the figure above, we have a Turing Test 1 (T1) which is trained to discriminate the fake image (SR) from the original image (HR). The Turing Test 2 (T2) is trained to perform the same discrimination only on the craters. Likewise Turing Test 3 (T3) is trained to discriminate the hills in the lunar surface. We detect the hills and craters from the OHRC images by manual annotation.

## Eval of SR images using Feature Comparison

Observing changes in lunar super-resolution images can provide valuable information about the geological and physical processes that have shaped the moon's surface over time. This can provide a better understanding of the moon's history and evolution, as well as help in planning for future missions to the moon. The high-resolution images can also reveal new features and details that were previously not visible, leading to new discoveries and scientific insights. We have built a variety of algorithms for comparison of physical features obtainable from the lunar images, before and after super-resolution. This conveys the improvement in the detection and analysis of features in the super-resolved images.

### Dynamic Thresholding Algorithm:
We have used a dynamic thresholding algorithm on the DEM data. We have made a histogram of the pixel values and have considered the top 2% of the pixels for identifying hills within the terrain data. Similarly, we have considered the bottom 2% of the pixels for identifying craters. 

### Clustering Algorithm:
We have clustered the craters together to identify and count the number of craters in an image. 



## Stitched Atlas

![Alt text](images/atlas_resized.png?raw=true "Figure shows the complete stitched lunar atlas")

We stitched all the avaialble TMC-2 data to form the complete stitched lunar atlas, which ranges from -180° to +180° in longitude and -90° to 90° in latitude as shown above.



<!-- ACKNOWLEDGEMENTS -->

## Acknowledgements
