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

A PyTorch-based image classifier using transfer learning with ResNet50 to classify food images as pizza, steak, or sushi.

The final ResNet50 model achieved **98.12% test accuracy** on the pizza/steak/sushi classification task.

- App URL: https://huggingface.co/spaces/rogerqiu77/foodvision-deep-learning-classifier
- Github Repo: https://github.com/rogerqiu7/foodvision-deep-learning-classifier

---

## 🎯 Features

- **Transfer Learning** with pretrained ResNet50 and EfficientNet-B2
- **MLflow Experiment Tracking** for model comparison and versioning
- **Gradio Web Interface** for real-time predictions
- **98.12% Test Accuracy** on food image classification
- **Hugging Face Spaces Deployment** for sharing the model as a web app

---

## 📊 Results

| Model | Test Accuracy | Parameters | Training Time |
|-------|---------------|------------|---------------|
| **ResNet50** | **98.12%** | 23.5M | 5 min (5 epochs) |
| EfficientNet-B2 | 95.06% | 7.7M | 6 min (5 epochs) |

The ResNet50 model performed best, so it was saved and deployed in the Gradio app.

---

## 🛠️ Tech Stack

- **Framework:** PyTorch, torchvision
- **Experiment Tracking:** MLflow
- **Deployment:** Gradio, Hugging Face Spaces
- **Image Processing:** Pillow
- **Hardware:** Tesla T4 GPU, Google Colab

---

## 📁 Project Structure

```text
foodvision/
├── FoodVision.ipynb       # Complete training and experiment pipeline
├── app.py                 # Gradio app used for deployment
├── models/
│   └── best_model.pth     # Saved trained model weights
├── examples/              # Sample images used in the Gradio demo
├── requirements.txt       # Python dependencies
└── README.md
````

---

## 📈 Training Pipeline

1. **Data:** Pizza/Steak/Sushi image dataset
2. **Preprocessing:** Resize images, apply augmentation, normalize using pretrained model transforms
3. **Model:** Use pretrained ResNet50 and EfficientNet-B2
4. **Transfer Learning:** Freeze most pretrained layers and replace the final classifier layer
5. **Training:** Train the new classifier head on pizza, steak, and sushi images
6. **Tracking:** Use MLflow to compare model performance
7. **Deployment:** Save the best model and deploy it with Gradio on Hugging Face Spaces


---

## What `FoodVision.ipynb` Does

`FoodVision.ipynb` is the training notebook for the project. It trains, compares, and saves the best food classification model before `app.py` uses it in the Gradio app.

High-level flow:

```text
Load images
→ Apply transforms
→ Create pretrained model
→ Replace final layer
→ Train model
→ Compare results
→ Save best model
````

---

### 1. Load and prepare the data

The notebook loads the pizza/steak/sushi image dataset.

The model learns to classify images into three classes:

```text
pizza
steak
sushi
```

Before training, images are resized, converted to tensors, normalized, and augmented.

Example:

```text
Original image → resized/normalized image tensor
```

This matters because pretrained models like ResNet50 expect images in a specific format.

---

### 2. Use transfer learning

Instead of training a CNN from scratch, the notebook uses pretrained models like:

```text
ResNet50
EfficientNet-B2
```

These models already learned useful image features from large datasets, such as edges, textures, shapes, and object patterns.

The notebook keeps those learned features and adapts the model to the food dataset.

Original model:

```text
Image → ResNet50 → 1000 ImageNet classes
```

FoodVision model:

```text
Image → ResNet50 → 3 classes: pizza, steak, sushi
```

---

### 3. Freeze layers and replace the classifier

Most pretrained layers are frozen so their learned features do not change much.

Then the final classification layer is replaced with a new layer that predicts only three classes.

Simple idea:

```text
Frozen backbone = feature extractor
New classifier head = learns pizza/steak/sushi
```

This makes training faster and works well with a smaller dataset.

---

### 4. Train and compare models

The notebook trains different models and compares their results.

It tracks:

```text
accuracy
loss
training time
model size
```

Example:

```text
ResNet50: higher accuracy
EfficientNet-B2: smaller model, slightly lower accuracy
```

The best model is selected based on performance.

---

### 5. Track experiments with MLflow

MLflow is used to log experiment results.

Example:

```text
model = ResNet50
epochs = 5
learning rate = 0.001
test accuracy = 98.12%
```

This makes it easier to compare models and remember which settings worked best.

---

### 6. Save the best model

After training, the best model is saved as:

```text
models/best_model.pth
```

Then `app.py` loads that file and uses it for predictions in the Gradio app.

Simple flow:

```text
FoodVision.ipynb trains the model
→ saves best_model.pth
→ app.py loads it
→ Gradio app predicts pizza/steak/sushi
```

---

## How Transfer Learning Works

Transfer learning means using a model that has already learned useful image features from a large dataset.

For example, ResNet50 was pretrained on ImageNet, so it already knows how to detect general visual patterns like:

```text
edges
textures
shapes
colors
object parts
```

Instead of training a new model from scratch, this project reuses the pretrained model and only changes the final classification layer.

Original ResNet50:

```text
Image → ResNet50 → 1000 ImageNet classes
```

This project:

```text
Image → ResNet50 → 3 food classes
```

The three output classes are:

```text
pizza
steak
sushi
```

This makes training faster and more accurate because the model already understands basic image features.

---

## How `app.py` Works

`app.py` is the deployment file for the Gradio web app.

At a high level, it does this:

1. Loads the saved model checkpoint from `models/best_model.pth`
2. Recreates the same model architecture used during training
3. Loads the trained weights into the model
4. Preprocesses uploaded images
5. Runs the image through the model
6. Returns prediction probabilities for pizza, steak, and sushi
7. Displays the results in a Gradio interface

---

## Model Loading

The app first loads the saved checkpoint:

```python
checkpoint = torch.load("models/best_model.pth", map_location=device)
```

The checkpoint stores important training information, including:

```text
model_name
class_names
test_accuracy
model_state_dict
```

Then the app recreates the model architecture:

```python
model, inference_transform = create_model(model_name, num_classes=len(class_names))
```

This is important because PyTorch needs the model structure before it can load the saved weights.

Simple example:

```text
Step 1: Create empty ResNet50 structure
Step 2: Load trained weights into that structure
Step 3: Use the model for predictions
```

---

## Prediction Flow

When a user uploads an image, the `predict()` function runs.

The flow is:

```text
Uploaded image
→ Convert to PIL image if needed
→ Apply model transforms
→ Add batch dimension
→ Send image through model
→ Apply softmax
→ Return class probabilities
```

Example output:

```text
pizza: 0.96
steak: 0.03
sushi: 0.01
```

The highest probability is the model’s predicted class.

---

## Why Softmax Is Used

The model outputs raw scores called logits.

Softmax converts those raw scores into probabilities.

Example:

```text
Raw model output:
[5.2, 1.1, 0.3]

After softmax:
pizza: 0.96
steak: 0.03
sushi: 0.01
```

This makes the output easier to understand in the Gradio app.

---

## Gradio Web App

Gradio is used to create a simple web interface.

The app lets users:

1. Upload an image
2. Run the model
3. See the top predictions
4. Try example images

In `app.py`, the Gradio app is created with:

```python
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
)
```

This means:

```text
Input: image
Function: predict()
Output: class probabilities
```

The app also includes example images, so users can test the model without uploading their own images.

---

## Hugging Face Spaces Deployment

This project is deployed using Hugging Face Spaces.

The top section of the README tells Hugging Face how to run the app:

```yaml
sdk: gradio
app_file: app.py
```

This means Hugging Face will:

1. Install the dependencies from `requirements.txt`
2. Look for `app.py`
3. Run the Gradio app
4. Host the app as a public web demo

---

## Requirements

The main dependencies are:

```text
torch
torchvision
gradio
Pillow
```

These are listed in `requirements.txt`.

---

## Example Use Cases

Example questions this project answers visually:

```text
Is this image pizza, steak, or sushi?
```

```text
How confident is the model in its prediction?
```

```text
Can a pretrained CNN classify food images with a small dataset?
```

---

## 🎓 Key Takeaways

* Transfer learning reduces training time from hours to minutes
* ResNet50 performed better than EfficientNet-B2 on this dataset
* Freezing pretrained layers helps keep useful image features
* Replacing the final layer adapts the model to a new task
* MLflow makes it easier to compare experiments
* Gradio makes it easy to turn a model into an interactive web app
* Hugging Face Spaces makes the app easy to share and demo

---

## Summary

This project shows how to train and deploy a simple deep learning image classifier.

The main idea is:

```text
Image → Pretrained CNN → Custom classifier → Food prediction
```

Instead of training a model from scratch, the project uses transfer learning to quickly build a high-performing classifier for pizza, steak, and sushi images.

```
```
