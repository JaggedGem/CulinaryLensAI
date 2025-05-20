# Optimized Food-101 Classification System

This project provides a complete, optimized system for training high-accuracy deep learning models on the Food-101 dataset. The system features CUDA acceleration, mixed-precision training, automatic resume capabilities, and real-time visual indicators of training progress.

## Features

- **High-Performance Training**:
  - CUDA acceleration with mixed-precision
  - Optimized data loading and preprocessing
  - Advanced data augmentation strategies
  - Support for multiple model architectures
  - Learning rate schedulers and adaptive optimizers

- **Resilient Training Process**:
  - Automatically save progress at regular intervals
  - Gracefully handle interruptions (Ctrl+C) with checkpoint saving
  - Resume training from where it left off
  - Real-time visual progress indicators

- **Visual Monitoring**:
  - TensorBoard integration for real-time monitoring
  - Training history plots automatically generated
  - Detailed performance metrics and visualizations

- **Interactive Demo**:
  - User-friendly web interface for testing the model
  - Real-time predictions with confidence scores
  - Camera integration for live testing

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/food101-classification.git
   cd food101-classification
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Training the Model

Run the optimized training pipeline:

```bash
python train_optimized.py
```

This will:
1. Download and prepare the Food-101 dataset (~5GB)
2. Train an optimized EfficientNet-B0 model with CUDA acceleration
3. Save checkpoints, training history, and evaluation metrics
4. Create a ready-to-use model for inference

### Training Options

- `--model`: Model architecture (choices: resnet18, resnet34, resnet50, densenet121, efficientnet_b0, efficientnet_b2)
- `--batch-size`: Batch size for training
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--device`: Device to use (cuda or cpu)
- `--skip-dataset`: Skip dataset preparation
- `--auto-resume`: Automatically resume from the latest checkpoint (default: True)

Example with custom settings:
```bash
python train_optimized.py --model efficientnet_b2 --batch-size 32 --epochs 100 --lr 0.0003
```

### Resuming Interrupted Training

If training is interrupted (power failure, Ctrl+C, etc.), you can easily resume:

```bash
python resume_training.py
```

This will automatically find the latest checkpoint and resume training with the same parameters.

To override specific parameters:
```bash
python resume_training.py --epochs 200 --lr 0.0001
```

### Interactive Demo

After training, launch the interactive demo:

```bash
python food101_demo_app.py
```

Or with the training script:
```bash
python train_optimized.py --skip-training --demo
```

## Model Performance

With the default settings (EfficientNet-B0, 50 epochs), you can expect:
- Top-1 accuracy: ~85-87%
- Top-5 accuracy: ~96-97%

For best accuracy (may take longer to train):
```bash
python train_optimized.py --model efficientnet_b2 --batch-size 32 --epochs 100
```

## Advanced Features

### Mixed Precision Training

Mixed precision training is enabled by default to accelerate training on compatible GPUs while maintaining accuracy.

### Real-time Monitoring

TensorBoard is enabled by default for real-time monitoring of training:

```bash
tensorboard --logdir ./runs/food101/tensorboard
```

### Automatic Optimization

The optimized training pipeline includes:
- Learning rate scheduling (Cosine Annealing by default)
- Label smoothing for better generalization
- Dropout regularization
- Advanced data augmentation

## Visualization and Evaluation

After training completes, you'll find in the output directory:
- Training history plots
- Confusion matrix visualization
- Per-class accuracy analysis
- Classification reports

## License

This project is licensed under the MIT License.

## Acknowledgments

- The Food-101 dataset is provided by ETH Zurich
- Based on PyTorch and torchvision frameworks
