# AI-Based Mechanical CAD Image Difference Detection

## Overview

The **AI-Based Mechanical CAD Image Difference Detection** project is a Flask-based web application developed to compare two Mechanical CAD images and identify engineering design changes using traditional Computer Vision techniques. The system helps engineers quickly detect modifications between two versions of a CAD drawing without manual inspection.

The application accepts two CAD images (PNG, JPG, or JPEG), automatically aligns them using **ORB Feature Detection** and **Homography Transformation**, and performs image comparison using **Structural Similarity Index (SSIM)** and **Absolute Difference** techniques. It then applies **Canny Edge Detection**, **Morphological Operations**, and **Contour Detection** to identify meaningful design changes while reducing image noise.

The detected changes are highlighted with bounding boxes, and the application generates useful statistics such as the SSIM score, number of changed regions, changed pixels, percentage of changed area, and the largest detected change. A rule-based summary is also created to provide a simple description of the detected modifications.

The project is built entirely with **Python**, **Flask**, **OpenCV**, **NumPy**, **scikit-image**, **HTML5**, **CSS3**, and **Bootstrap 5**. It runs locally without requiring any cloud services, databases, machine learning models, or external APIs, making it lightweight, easy to install, and suitable for educational as well as industrial prototype applications.

## Features

* Compare two Mechanical CAD images
* Automatic image alignment using ORB and Homography
* Detect small engineering modifications
* Generate difference mask and highlighted output
* Display engineering statistics
* Produce a rule-based change summary
* Responsive Bootstrap-based user interface
* Offline execution with no AI or cloud dependency

This project demonstrates how classical image processing techniques can be effectively used to automate CAD drawing comparison and improve engineering design review workflows.
