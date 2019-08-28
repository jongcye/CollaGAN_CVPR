# Collaborative Generative Adversarial Networks 
Tensorflow implementation of Collaborative GAN ( CVPR2019, ORAL).

The main codes have two parts: one is Collaborative Generative Adversarial Networks for facial expression imputation problem.
The Collaborative GAN is a deep learning model for missing image data imputation (Dongwook Lee et al. CVPR 2019). The concept for the missing image imputation is applied for facial expression imputation problem and this is the implementation of that using tensorflow.

This repository provides a tensorflow implementation of CollaGAN for missing MR contrast imputation as described in the paper:
> CollaGAN: Collaborative GAN for missing  image data imputation.
> Dongwook Lee, Junyoung Kim, Won-Jin Moon, Jong Chul Ye 
> [[Paper]](CVPR2019, oral)

## OS
The package development version is tested on Linux operating systems. The developmental version of the package has been tested on the following systems:
Linux: Ubuntu 16.04

## Requirements
The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
tensorflow 		  1.10.1
tqdm			  4.28.1
numpy			  1.14.5
scipy			  1.1.0
argparse		  1.1
logging 	 	  0.5.1.2
ipdb 			  0.11
cv2 			  3.4.3
```
## Datasets
[Radboud Faces Database]
The RaFD is a high quality faces database, which contain pictures of eight emotional expressions.
Langner, O., Dotsch, R., Bijlstra, G., Wigboldus, D.H.J., Hawk, S.T., & van Knippenberg, A. (2010). Presentation and validation of the Radboud Faces Database. Cognition & Emotion, 24(8), 1377—1388. DOI: 10.1080/02699930903485076

## Main train files
```
train.py
```
These files are handled by the `scripts/run.sh` file with following commands:
```
sh scripts/run.sh
```

## Input and output options
The explanation of the input and output options for CollaGAN model and Segmentation model for training are introduced in following files, respectively:
```
options/facial_expression.py
```
The input dropout technique is implemented in the 'data/rafd8.py'.
