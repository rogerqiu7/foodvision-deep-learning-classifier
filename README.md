---
title: FoodVision Classifier
emoji: 🍕
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# FoodVision: Deep Learning Food Classifier
- App URL: https://huggingface.co/spaces/rogerqiu77/foodvision-deep-learning-classifier
- Github Repo: https://github.com/rogerqiu7/foodvision-deep-learning-classifier

A PyTorch-based image classifier using transfer learning (ResNet50) achieving 98% accuracy on pizza/steak/sushi classification.

## 🎯 Features

- **Transfer Learning** with pretrained ResNet50 and EfficientNet-B2
- **MLflow Experiment Tracking** for model comparison and versioning
- **Gradio Web Interface** for real-time predictions
- **98.12% Test Accuracy** on food image classification

## 🚀 [Try the Live Demo](https://huggingface.co/spaces/YOUR_USERNAME/foodvision-classifier)

## 📊 Results

| Model | Test Accuracy | Parameters | Training Time |
|-------|---------------|------------|---------------|
| **ResNet50** | **98.12%** | 23.5M | 5 min (5 epochs) |
| EfficientNet-B2 | 95.06% | 7.7M | 6 min (5 epochs) |

## 🛠️ Tech Stack

- **Framework:** PyTorch, torchvision
- **Experiment Tracking:** MLflow
- **Deployment:** Gradio, Hugging Face Spaces
- **Hardware:** Tesla T4 GPU (Google Colab)

## 📁 Project Structure
```
foodvision/
├── foodvision.ipynb       # Complete training pipeline
├── app.py                 # Gradio deployment
├── models/
│   └── best_model.pth     # Trained weights (98.12% accuracy)
├── examples/              # Sample images for demo
└── requirements.txt
```

## 📈 Training Pipeline

1. **Data:** Pizza/Steak/Sushi dataset (20% subset for quick prototyping)
2. **Preprocessing:** Resize to 224×224, TrivialAugmentWide, ImageNet normalization
3. **Model:** Pretrained ResNet50 with frozen backbone, custom classifier head
4. **Training:** 5 epochs with Adam optimizer (lr=0.001, batch_size=32)
5. **Tracking:** MLflow logged all experiments for comparison
6. **Deployment:** Best model deployed to Hugging Face Spaces

## 🎓 Key Takeaways

- Transfer learning reduces training from hours to minutes
- Data augmentation significantly improves generalization
- MLflow streamlines experiment comparison across architectures
- Gradio enables rapid prototyping and deployment