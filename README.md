---
title: FoodVision Classifier
emoji: 🍕
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# FoodVision: Deep Learning Food Classifier

A PyTorch-based image classifier using transfer learning (ResNet50) achieving 98% accuracy on pizza/steak/sushi classification.

## Features
- Transfer learning with pretrained ResNet50
- Real-time inference
- 98.12% test accuracy

## Usage
Upload an image of pizza, steak, or sushi to get predictions!
```

---

### **Step 2: Prepare Your Files Locally**

Create this structure:
```
foodvision/
├── app.py
├── requirements.txt
├── README.md
└── models/
    └── best_model.pth