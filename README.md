# 🛣️ Road Lane Detection Using Computer Vision

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?style=flat&logo=numpy&logoColor=white)

A **computer vision pipeline** for detecting lane lines on roads, built as part of the Self-Driving Car Engineer Nanodegree. Uses Canny edge detection, Hough transforms, and region-of-interest masking to identify and overlay lane markings on images and video.

## ✨ Features

- 🔍 **Grayscale conversion** and **Gaussian blur** for noise reduction
- 📐 **Canny edge detection** for identifying lane boundaries
- 🎯 **Region-of-interest masking** — dynamic triangular ROI based on image dimensions
- 📏 **Hough line transform** for detecting line segments
- 🧮 **Slope-based lane separation** — distinguishes left vs. right lane markings
- 📈 **Line averaging & extrapolation** — smooth, full-length lane overlay
- 🎬 **Video processing** — frame-by-frame lane detection on MP4 files using MoviePy
- 🎨 **Color-based lane detection** — HLS color space for yellow and white lane identification

## 🔄 Pipeline Overview

```
Input Image/Frame
  → Grayscale conversion
  → Gaussian blur (kernel=11)
  → Canny edge detection (50/150 thresholds)
  → Region-of-interest masking
  → Hough line transform
  → Slope filtering & lane averaging
  → Weighted overlay on original image
```

## 🛠️ Tech Stack

- **Python 3** — core language
- **OpenCV** — image processing and computer vision
- **NumPy** — numerical operations
- **Matplotlib** — visualization
- **MoviePy** — video processing

## 🚀 Getting Started

### Prerequisites

```bash
pip install opencv-python numpy matplotlib moviepy
```

### Run the Notebook

```bash
jupyter notebook "Finding Lane Lines- CARND-Term-1- Submission.ipynb"
```

The notebook walks through the full pipeline with test images and video processing.

## 📁 Project Structure

```
Road-Lane-Detection-Using-Computer-Vision/
├── Finding Lane Lines- CARND-Term-1- Submission.ipynb   # Main notebook
├── output_4_1.png                                        # Sample output
├── output_29_*.png                                       # Test image results
└── README.md
```

## 📸 Sample Results

The pipeline processes multiple test images (solid white, solid yellow, curves) and three video clips (white right, yellow left, challenge) with lane overlay rendering.

## 👤 Author

**Sanjay Santhanam**
