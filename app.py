import os
import shutil
import random
from pathlib import Path

import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
import torchvision

# Your create_model function (copy from your notebook)
def create_model(model_name: str, num_classes: int = 3, seed: int = 42):
    """Creates a pretrained model with custom classifier."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    if model_name == "efficientnet_b2":
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        model = torchvision.models.efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        
        for param in model.features.parameters():
            param.requires_grad = False
            
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3, inplace=True),
            torch.nn.Linear(in_features, num_classes),
        )
        
    elif model_name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        model = torchvision.models.resnet50(weights=weights)
        in_features = model.fc.in_features
        
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
            
        model.fc = torch.nn.Linear(in_features, num_classes)
    
    else:
        raise ValueError(f"Model '{model_name}' not supported")
    
    transforms_obj = weights.transforms()
    return model, transforms_obj


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
checkpoint = torch.load("models/best_model.pth", map_location=device)
model_name = checkpoint["model_name"]
class_names = checkpoint["class_names"]
test_accuracy = checkpoint["test_acc"]

model, inference_transform = create_model(model_name, num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# Prediction function
def predict(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    img_tensor = inference_transform(image).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
    
    return {class_names[i]: float(probs[0][i]) for i in range(len(class_names))}

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="🍕🥩🍣 FoodVision Classifier",
    description=f"Upload **pizza**, **steak**, or **sushi** images!\\n\\n**Model:** {model_name} | **Accuracy:** {test_accuracy:.2%}",
    theme="soft",
    examples=[
        ["examples/pizza.jpg"],
        ["examples/steak.jpg"],
        ["examples/sushi.jpg"],
        ["examples/pizza2.jpg"],
        ["examples/steak2.jpg"],
        ["examples/sushi2.jpg"],
        ["examples/pizza3.jpg"],
        ["examples/steak3.jpg"],
        ["examples/sushi3.jpg"]
    ],
    examples_per_page=3,
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()