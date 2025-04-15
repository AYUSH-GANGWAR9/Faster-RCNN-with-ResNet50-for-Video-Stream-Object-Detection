# 🎯 Faster R-CNN with ResNet50 for Video Stream Object Detection

## 📋 Overview

This repository contains implementation of a Faster Region-Convolutional Neural Network (FRCNN) with ResNet50 backbone for object detection in video streams. The model addresses challenges in video object detection including poor frame quality, pose changes, occlusions, video defocusing, and motion blur.

## 🔍 Problem Statement

- Video stream object detection is challenged by poor quality of frames (pose changes, occlusions, defocusing, motion blur)
- Frame-by-frame detection in video streams is computationally intensive
- Closely arranged frames in video streams make object detection more complex

## 🎯 Objectives

- Improve robustness in detecting objects in video streams by developing algorithms that effectively handle quality challenges
- Enhance computational efficiency with a simplified approach for processing video frames
- Reduce false detection rates in video stream object detection

## 🛠️ Methodology

The model architecture combines:

1. **Convolutional Neural Network (CNN)** - Processes input video frames
2. **Region Proposal Network (RPN)** - Generates object proposals from feature maps
3. **ROI Pooling Layer** - Standardizes the sizes of varying proposals
4. **ResNet50 Backbone** - Extracts deep semantic features while addressing vanishing gradient problems
5. **Fully Connected Layers** - Classify objects based on extracted features

### 📊 Model Architecture

```
Input Video Frame → ResNet50 Feature Extraction → Region Proposal Network → 
ROI Pooling → Segmentation Mask → Fully Connected Layer → Object Classification
```

## 📈 Performance

The model was evaluated on the DAVIS dataset and achieved:

| Metric | Score |
|--------|-------|
| F-Measure | 0.893 |
| S-Measure | 0.859 |
| Mean Absolute Error (MAE) | 0.016 |

### Comparative Analysis

| Method | F-Measure | MAE |
|--------|-----------|-----|
| CSAtt-ConvLSTM | 0.817 | 0.024 |
| DEEP-GCN | 0.844 | 0.031 |
| CFCN-MA | 0.867 | 0.020 |
| **FRCNN-ResNet50 (Ours)** | **0.893** | **0.016** |

## 📊 Dataset

The model was trained and tested on the Densely Annotated Video Segmentation (DAVIS) dataset:
- 50 videos totaling 3,455 frames
- Each video contains approximately 70 frames (854×480 pixels resolution)
- Includes binary ground truth annotations

## 🔧 Requirements

- MATLAB R2023a
- Intel Core i7 processor (or equivalent)
- 8GB RAM minimum

## 💡 Advantages of Our Approach

- Higher detection rate through ResNet50 feature extraction
- Lower false detection rates compared to existing models
- Effective handling of motion blur and occlusions in video streams
- Parallel implementation of ResNet50 with five fully connected layers to avoid false detections

## 👥 Contributors <br>

- Ayush gangwar
<br><hr>

## 🔜 Future Work

- Investigate model adaptations for specific object detection
- Improve model efficiency for detecting singular objects in video streams
- Enhance real-time performance for deployment in resource-constrained environments


## 📄 License

[MIT License](LICENSE)