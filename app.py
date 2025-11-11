import os
import shutil
import random
from pathlib import Path

import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
import torchvision

# create model method copied from notebook, used to create a model and trasnformation
def create_model(model_name: str, num_classes: int = 3, seed: int = 42):
    """
    Creates pretrained model with frozen layers + custom classifier.

    Transfer learning strategy:
    1. Load model pretrained on ImageNet (1000 classes)
    2. Freeze early layers (feature extraction remains unchanged)
    3. Replace final layer to output 3 classes (pizza/steak/sushi)
    4. Only train the new classifier head

    Args:
        model_name: "efficientnet_b2" or "resnet50"
        num_classes: Output classes (3 for this dataset)
        seed: Reproducibility for weight initialization

    Returns:
        (model, pretrained_transforms)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    if model_name == "efficientnet_b2":
        # load pretrained weights from ImageNet
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        model = torchvision.models.efficientnet_b2(weights=weights)

        # number of features coming out of backbone and into the final classification layer
        in_features = model.classifier[1].in_features
        
        # freeze all convolutional layers (pretrained features are good enough)
        for param in model.features.parameters():
            param.requires_grad = False
        
        # replace classifier head, final layer: 1408 → 3 classes
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


# setup device based on cuda availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load best model from previous training from notebook
checkpoint = torch.load("models/best_model.pth", map_location=device)
model_name = checkpoint["model_name"]
class_names = checkpoint["class_names"]
test_accuracy = checkpoint["test_acc"]

# create_model() builds ResNet50/EfficientNet with random weights
# We need the architecture to match before loading trained weights
model, inference_transform = create_model(model_name, num_classes=len(class_names))

# load trained weights into the model with random weights, checkpoint["model_state_dict"] contains all the learned parameters
model.load_state_dict(checkpoint["model_state_dict"])

# move to GPU/CPU
model = model.to(device)
model.eval()

# Prediction function
def predict(image):
    """
    Takes an image and returns class probabilities.

    Args:
        image: PIL Image or numpy array from Gradio

    Returns:
        Dictionary mapping class names to probabilities (0-1)
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # preprocess image and add batch dimension
    img_tensor = inference_transform(image).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
    
    # convert to dictionary of class names and probabilities
    return {class_names[i]: float(probs[0][i]) for i in range(len(class_names))}

# created model by best name, loaded training weights, used in predict function
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),

    # UI customization
    title="🍕🥩🍣 FoodVision Classifier",
    description=f"Upload **pizza**, **steak**, or **sushi** images!\\n\\n**Model:** {model_name} | **Accuracy:** {test_accuracy:.2%}",
    theme="soft",

    # Example images (clickable in UI)
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

# launch the app
if __name__ == "__main__":
    demo.launch()