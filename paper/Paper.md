# Adaptive Resonance Theory-Guided Novelty Detection for Efficient Sample Selection in Object Detection

## Abstract
Annotating large-scale object detection datasets is a major bottleneck in training high-performance models, especially when deploying to edge devices with limited storage and compute. To address this, we present a framework that integrates Adaptive Resonance Theory (ART) with YOLO-based object detection to enable efficient, real-time novelty detection. By extracting intermediate feature representations from the YOLO model’s hidden layers, our system employs an unsupervised ART-based clustering algorithm to identify and flag novel instances—data that significantly deviates from previously encountered distributions. This method reduces redundant data collection, highlights edge cases and underrepresented scenarios, and substantially lowers annotation demands without sacrificing model accuracy. Our approach is well-suited for on-device learning workflows and data prioritization in resource-constrained environments.

## Introduction

### Why object detection needs efficient annotation/data seleciton

### Challenges in current active learning or uncertainty-based approaches

### Motivation for using ART
Its unsupervised increamentation nature

### What you method does and key contributions


## Methodology
We are going to do three trials
- feature vector from the backbone layers
- feature vector from the penultimate layer before detection head
- feature vector from both
  
ART Models used
- Fuzzy ART
- Novelity is detected when match scores no longer meet a threshold

Inference Pipeline
TBD

## Experiments

### Datasets 
We will perform the experment by taking 10 percent of the coco dataset, training the yolo model on that subset. Then retraining it on the next 10 percent only selecting images which did not meet the threshold.

The control experment will have a random sample of the same amount of images that the ART module selected.

There are some inherent flaws in that this does not replicate a real production envirement, but it fits our budget.

### Baselines 
(random sampling, entropy-based sampling, clustering without ART).

### Metrics
mAP vs. labeled volume
Novelity detection quality
ART cluster interpretability

### Related work
Detection of Data Drift and Outliers Affecting Machine Learning Model
Performance Over Time
https://arxiv.org/pdf/2012.09258

ODIN: Automated Drift Detection and Recovery in Video Analytics
https://arxiv.org/abs/2009.05440

Data Models for Dataset Drift Controls in Machine Learning With Optical Images
https://arxiv.org/abs/2012.09258


Active learning for object detection
ART in machine learning
Novelty/outlier detection in deep learning
Data selection/sample efficiency for edge devices