import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import ViTImageProcessor
from model import MultiTaskViT

# Load checkpoint
checkpoint = torch.load("vit_best_model.pth", map_location="cpu")
label_mappings = checkpoint["label_mappings"]
num_labels = {col: len(classes) for col, classes in label_mappings.items()}

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiTaskViT("google/vit-base-patch16-224-in21k", num_labels)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Streamlit App
st.title("🖼️ Fashion Multi-Attribute Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor)

    # Display predictions
    st.subheader("Predictions:")
    for label_name, logits in outputs.items():
        pred_idx = torch.argmax(logits, dim=1).item()
        pred_label = label_mappings[label_name][pred_idx]
        st.success(f"🎯 Predicted {label_name}: **{pred_label}**")
