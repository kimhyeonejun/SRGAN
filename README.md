# SRGAN
Implementation of SRGAN from Photo-realistic Single Image Super-resolution Using a Generative Adversarial Network
This work provides the implementation of SRGAN, a super-resolution model using Generative Adversarial Network, presented in 2017. I have used this paper to build SRGAN model. The model is implemented using pytorch. The quality of generated model is better than the original paper. DIV2K dataset is used as training dataset and Set 14 is used as test dataset. The comparision of this image and original image is given below.

|`     <LR>                 <SRGAN>                 <HR>      `|
|----------------------------|
|![image](https://github.com/kimhyeonejun/SRGAN/assets/103301952/72359723-bcae-4bcd-8786-aba0402bd0da)|

The comparision takes place with 4-upscaling images. Now, I compare this image with that given in the original paper.
| `<Mine>`        |        `<PAPER>`   |
|----------------------------|-------|
|![image](https://github.com/kimhyeonejun/SRGAN/assets/103301952/d2ad3e2c-fd90-4950-a995-aa9fcf168cc6)|![image](https://github.com/kimhyeonejun/SRGAN/assets/103301952/cfc1e859-5e59-4029-a574-fa8a44ec7139)|

Lpips index is a method used for measuring the quality of image, invented after the publication of original paper. In the origianl paper, MOS (Mean Opinion Score), which is subjective, is used for quality assessment. Instead, I make reference to more objective index, lpips, and compare it, indicated in this paper, SIR-SRGAN: Super-Resolution Generative Adversarial Network with Self-Interpolation Ranker, with my model.

| `<Mine>`  | `<Paper>`   |
|------------|----------------|
| lpips: 0.2279 | lpips : 0.2712 |

Note that my measurement is done from importing lpips library in python. 
