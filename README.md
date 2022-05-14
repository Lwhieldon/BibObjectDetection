<p align="center">
<img src="https://github.com/Lwhieldon/BibObjectDetection/blob/main/notebooks+utils+data/BibDetectorSample.jpeg?raw=true" height=400 />
</p>

# Natural Scene Race Bib Number Detection
Using machine learning algorithms, including OpenCV, NIVIDIA's cuDNN, &amp; Darknet's YOLOv4 to detect numbers on racing bibs found in natural image scene. 
<br>
<br>
## Overview and Background

Racing Bib Number (RBN) detection and recognition contains the interesting tasks of both location of bib attached to a person in a natural scene & inferring the text detection on the bib. Since text recognition is required, additional steps of training is needed: Finding the area of the bib on a person and then inferring the numbers on the bib thereafter. This project uses the research & experience from prior implementations to apply a working model to detect race bib numbers in both images & video. 

This repo investigates the use of Convolutional Neural Networks (CNN) and specifically, <a href=https://developer.nvidia.com/cudnn>NVIDIA's cuDNN</a> & <a href=https://github.com/AlexeyAB/darknet>Darknet's You Only Look Once ver. 4 (YOLOv4),</a> to detect Racing Bib Numbers (RBNR) in a natural image scene. Leveraging publically available & labeled datasets from previous research  (please see reference section below for addt'l information), I achieve a mean average precision (mAP) on the following:

- 96% mAP on Street View House Number dataset training
- 99% mAP on Bib Detection in Natural Scene dataset training

For future work, I would like to continue to train the model with datasets from both the Street View House Numbers & more runner images with racing bibs. Additionally, I would like to transfer the learning weights to edge compute hardware (i.e. NVIDIA's Jetson Nano or Raspberr Pi) to display a use case for edge computing with live streaming video capture.

## Sample Video Output of Detection

![gif](https://github.com/Lwhieldon/BibObjectDetection/blob/main/notebooks+utils+data/marathon_output.gif)

## Data Details

- <a href=http://ufldl.stanford.edu/housenumbers>Street View House Numbers (SVHN) Dataset</a>: This dataset was curated for the NIPS Workshop on Deep Learning and Unsupervised Feature Learning from 2011. For additional information regarding the dataset, please contact streetviewhousenumbers@gmail.com.
- <a href=https://people.csail.mit.edu/talidekel/RBNR.html>Racing Bib Number Recognition Dataset</a>: Supporting dataset for the <a href=https://people.csail.mit.edu/talidekel/papers/RBNR.pdf>published paper</a> leveraging a different approach than we cover here.
<br>
<br>

## Table of Contents
```
bib-detector
|__ notebooks-utils-data
|   |__ 01- Prepocessing & Training SVHN YOLOv4-tiny Darknet.ipynb  
|   |__ 02 - Digit Detector Validation SVHN Training Only.ipynb 
|   |__ 03 - Preprocessing Racing Bib Numbers (RBNR) Datasets.ipynb
|   |__ 04 - Run Yolov4 Tiny on RBNR Data.ipynb
|   |__ 05 - Bib Detection Validation & Demo.ipynb
|   |__ utils.py
|   |__ VIDEO0433.mp4
|   |__ output_marathon.mp4
|   |__ BibDetectorSample.jpeg
weights-classes
|   |__ SVHN_obj.names
|   |__ RBNR_obj.data 
|   |__ SVHN_custom-yolov4-tiny-detector.cfg
|   |__ SVHN_custom-yolov4-tiny-detector_best.weights
|   |__ RBNR_custom-yolov4-tiny-detector.cfg
|   |__ RBNR_custom-yolov4-tiny-detector_best.weights
README.md
```
<br>
<br>

## References
Note also that notebooks were created into Google Collaboratory to take advantage of GPU throughput to speed the training of the CNN.
<br>
<pre>
Contributors : <a href=https://github.com/Lwhieldon>Lee Whieldon</a>
</pre>

<pre>
Languages    : Python
Tools/IDE    : Google Colab, Darknet, cuDNN
Libraries    : h5py, numpy, cv2, os, matplotlib.pyplot, scipy.io, pandas, imgaug
</pre>

<pre>
Assignment Submitted     : May 2022
</pre>

