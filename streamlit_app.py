import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import pipeline

# Load ONNX Model
MODEL_PATH = "/workspaces/blank-app/yolo_model.onnx"
session = ort.InferenceSession(MODEL_PATH)

# Fashion categories from dataset.yaml
CATEGORY_NAMES = ['Pakaian', 'Aksesoris', 'Alas kaki', 'Perawatan pribadi', 'Item gratis', 'Barang olahraga', 'Rumah']

# Load Caption Generator (Indonesian)
caption_generator = pipeline("text-generation", model="cahya/gpt2-small-indonesian-522M")

st.title("🛍️ AI Deteksi Produk Fashion")
st.write("Unggah foto fashion, dan dapatkan deskripsi produk serta caption media sosial!")

# Upload Image
uploaded_file = st.file_uploader("Unggah foto fashion", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocess image for YOLO ONNX
    img_resized = img.resize((256, 256))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)

    # Run YOLO ONNX Inference
    inputs = {session.get_inputs()[0].name: img_array}
    outputs = session.run(None, inputs)

    # Extract YOLO predictions (dummy logic for now)
    detected_labels = [CATEGORY_NAMES[i] for i in range(len(CATEGORY_NAMES)) if np.random.rand() > 0.5]  # Replace with actual processing

    # Generate Social Media Caption
    if detected_labels:
        caption_prompt = f"Foto ini menunjukkan {', '.join(detected_labels)}. Cocok untuk gaya sehari-hari!"
        caption = caption_generator(caption_prompt, max_length=30)[0]['generated_text']
    else:
        caption = "Tidak ada produk yang terdeteksi."

    st.subheader("🛍️ Produk yang Ditemukan")
    st.write(", ".join(detected_labels) if detected_labels else "Tidak ada produk yang terdeteksi.")

    st.subheader("📲 Caption untuk Media Sosial")
    st.write(caption)
