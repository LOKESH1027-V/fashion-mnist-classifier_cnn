
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import random
 
# ── 1. Model Class ──────────────────────────────────────────────
class FashionMNISTModelV3(nn.Module):
    def __init__(self, input: int, hidden: int, out: int):
        super().__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=input, out_channels=hidden,
                      kernel_size=(2, 2), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden, out_channels=hidden,
                      kernel_size=(2, 2), padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden, out_channels=hidden,
                      kernel_size=(2, 2), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden, out_channels=hidden,
                      kernel_size=(2, 2), padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.Linear_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden * 8 * 8, out_features=out)
        )
 
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.Linear_layer(x)
        return x
 
# ── 2. Class Names ──────────────────────────────────────────────
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_emojis = ['👕', '👖', '🧥', '👗', '🧥', '👡', '👔', '👟', '👜', '👢']
 
# ── 3. Load Model & Dataset ─────────────────────────────────────
@st.cache_resource
def load_model():
    model = FashionMNISTModelV3(input=1, hidden=10, out=10)
    model.load_state_dict(torch.load("fashionmnist_cnn.pth",
                          map_location=torch.device("cpu")))
    model.eval()
    return model
 
@st.cache_data
def load_dataset():
    test = datasets.FashionMNIST(
        root="data", train=False, download=True,
        transform=transforms.ToTensor()
    )
    return test
 
model = load_model()
test_data = load_dataset()
 
# ── 4. Transform for uploaded images ───────────────────────────
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])
 
# ── 5. Prediction helper ────────────────────────────────────────
def predict(img_tensor):
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item() * 100
    return predicted_idx, confidence, probabilities
 
def show_results(col, predicted_idx, confidence, probabilities, true_label=None):
    with col:
        st.subheader("Prediction")
        if true_label is not None:
            if predicted_idx == true_label:
                st.success(f"✅ {class_emojis[predicted_idx]} **{class_names[predicted_idx]}**")
                st.write("**Correct prediction!** 🎉")
            else:
                st.error(f"❌ {class_emojis[predicted_idx]} **{class_names[predicted_idx]}**")
                st.write(f"**Wrong!** Actual: {class_emojis[true_label]} {class_names[true_label]}")
        else:
            st.success(f"{class_emojis[predicted_idx]} **{class_names[predicted_idx]}**")
 
        st.write(f"Confidence: **{confidence:.2f}%**")
 
        st.subheader("Top 3 Predictions")
        top3 = torch.topk(probabilities, 3)
        for i in range(3):
            i_idx = top3.indices[0][i].item()
            prob = top3.values[0][i].item() * 100
            st.write(f"{class_emojis[i_idx]} {class_names[i_idx]}: **{prob:.2f}%**")
 
# ── 6. UI ───────────────────────────────────────────────────────
st.title("👗 Fashion MNIST Classifier")
st.write("Classify clothing items using a CNN model built with PyTorch!")
st.markdown("---")
 
# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.write("CNN model built with PyTorch — **89.88% accuracy!**")
st.sidebar.write("**Model:** FashionMNISTModelV3")
st.sidebar.write("**Framework:** PyTorch")
st.sidebar.write("**Dataset:** Fashion MNIST (10,000 test images)")
st.sidebar.markdown("---")
st.sidebar.write("**Classes:**")
for emoji, name in zip(class_emojis, class_names):
    st.sidebar.write(f"{emoji} {name}")
 
# Tabs for two modes
tab1, tab2 = st.tabs(["🎲 Random MNIST Image", "📁 Upload Your Image"])
 
# ── Tab 1: Random MNIST Image ───────────────────────────────────
with tab1:
    st.subheader("Test with Fashion MNIST Dataset")
    st.write("Click the button to pick a random image from the test dataset!")
 
    if st.button("🎲 Pick Random Image"):
        idx = random.randint(0, len(test_data) - 1)
        img_tensor, true_label = test_data[idx]
 
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Image")
            img_np = img_tensor.squeeze().numpy()
            fig, ax = plt.subplots()
            ax.imshow(img_np, cmap="gray")
            ax.axis("off")
            st.pyplot(fig)
            st.write(f"**Actual:** {class_emojis[true_label]} {class_names[true_label]}")
 
        predicted_idx, confidence, probabilities = predict(img_tensor)
        show_results(col2, predicted_idx, confidence, probabilities, true_label)
 
# ── Tab 2: Upload Image ─────────────────────────────────────────
with tab2:
    st.subheader("Upload Your Own Image")
    st.write("Upload a clothing image — works best with simple images on plain background!")
    st.info("⚠️ Model is trained on grayscale 28x28 images. Color images will be converted automatically.")
 
    uploaded_file = st.file_uploader(
        "Upload a clothing image",
        type=['png', 'jpg', 'jpeg']
    )
 
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
 
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, width=200)
 
        img_tensor = transform(image)
        predicted_idx, confidence, probabilities = predict(img_tensor)
        show_results(col2, predicted_idx, confidence, probabilities)
 
st.markdown("---")
