# Food-101 Classification System

This repository contains a complete system for training and evaluating deep learning models on the Food-101 dataset, a collection of 101,000 food images across 101 different categories.

## Table of Contents

1. [Overview](#overview)
2. [Setup and Installation](#setup-and-installation)
3. [Dataset](#dataset)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Inference](#inference)
7. [Demo App](#demo-app)
8. [Results](#results)

## Overview

The Food-101 Classification System allows you to:

- Download and prepare the Food-101 dataset
- Train deep learning models to classify food images
- Evaluate model performance
- Make predictions on new images
- Run an interactive demo app

## Setup and Installation

### Prerequisites

- Python 3.8 or later
- CUDA-capable GPU recommended (6GB+ VRAM)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/food101-classification.git
cd food101-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The Food-101 dataset contains 101,000 images of food divided into 101 categories, with 1,000 images per category. For each class, 250 manually reviewed test images are provided, while the remaining 750 training images contain some noise.

### Dataset Preparation

To download and prepare the dataset:

```bash
python prepare_food101.py
```

This script:
- Downloads the Food-101 dataset (~5GB)
- Extracts the dataset
- Organizes it into a structure suitable for training

The dataset will be organized in the following structure:
```
food-101/
├── train/
│   ├── apple_pie/
│   ├── baby_back_ribs/
│   └── ...
└── test/
    ├── apple_pie/
    ├── baby_back_ribs/
    └── ...
```

## Training

You can train a model on the Food-101 dataset using:

```bash
python train_food101.py
```

### Training Options

- `--data-dir`: Directory for the dataset (default: ./food-101)
- `--batch-size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 20)
- `--model`: Model architecture - resnet18, resnet50, densenet121, efficientnet_b0 (default: resnet50)
- `--lr`: Learning rate (default: 0.001)
- `--output-dir`: Directory to save results (default: ./runs/food101)
- `--device`: Device to use - cuda or cpu (default: cuda if available)
- `--skip-dataset`: Skip dataset preparation
- `--skip-training`: Skip model training (useful for just running the demo)
- `--demo`: Launch demo app after training
- `--share`: Create a public link for the demo app

### Advanced Training

For detailed control over the training process, you can use:

```bash
python food101_classifier.py --data-dir ./food-101 --epochs 50 --model efficientnet_b0
```

## Evaluation

Evaluation is automatically performed after training. The results include:

- Test accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score for each class)

Results are saved to the output directory.

## Inference

To make predictions on new images:

```bash
python food101_inference.py --image path/to/your/food_image.jpg --model-path ./runs/food101/best_model.pth
```

### Inference Options

- `--image`: Path to the input image (required)
- `--model-path`: Path to the model checkpoint
- `--class-names`: Path to the class names JSON file
- `--model-type`: Model architecture
- `--output`: Path to save visualization output
- `--top-k`: Show top K predictions (default: 5)

## Demo App

You can launch an interactive web app for food classification:

```bash
python food101_demo_app.py --model-path ./runs/food101/best_model.pth
```

This launches a Gradio interface where you can upload images and get real-time predictions.

### Demo Options

- `--model-path`: Path to the model checkpoint
- `--class-names`: Path to the class names JSON file
- `--model-type`: Model architecture
- `--share`: Create a public link for the app
- `--port`: Port to run the app on (default: 7860)

## Results

With a ResNet-50 model trained for 20 epochs, you can expect:

- Top-1 accuracy: ~80-85%
- Top-5 accuracy: ~95%

The results may vary depending on the training configuration and random initialization.

## References

- Bossard, L., Guillaumin, M., & Van Gool, L. (2014). Food-101 – Mining Discriminative Components with Random Forests. European Conference on Computer Vision.
