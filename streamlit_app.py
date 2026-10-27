import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import gdown
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from facenet_pytorch import MTCNN

st.title("🛡️ Deepfake Detection AI")
st.write("Upload a face image to evaluate its authenticity using EfficientNet-B0.")

device = torch.device("cpu")
CHECKPOINT_PATH = "best_deepfake_model.pt"
DRIVE_FILE_ID = "1IKlstbHz9bq7sVT_1ULIjuQGk6HKFhVS"

@st.cache_resource
def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        with st.spinner("Downloading model weights from Google Drive..."):
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, CHECKPOINT_PATH, quiet=False)

    class DeepfakeDetector(nn.Module):
        def __init__(self):
            super().__init__()
            weights = EfficientNet_B0_Weights.DEFAULT
            self.backbone = efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, 1)
            )

        def forward(self, x):
            return self.backbone(x)

    model = DeepfakeDetector().to(device)
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

model = load_model()
mtcnn = MTCNN(image_size=384, margin=20, keep_all=False, select_largest=True, post_process=False, device=device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Analyzing image..."):
        try:
            face_tensor = mtcnn(img)
            if face_tensor is not None:
                face_np = face_tensor.permute(1, 2, 0).cpu().numpy().astype('uint8')
                crop = Image.fromarray(face_np)
            else:
                crop = img.resize((384, 384))
        except Exception:
            crop = img.resize((384, 384))

        tensor_in = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(tensor_in).squeeze().item()
            prob_real = torch.sigmoid(torch.tensor(logit)).item()

        pred_label = "REAL" if prob_real >= 0.50 else "FAKE"
        conf_score = prob_real if pred_label == "REAL" else (1.0 - prob_real)

        st.success(f"**Prediction:** {pred_label}")
        st.info(f"**Confidence Score:** {conf_score * 100:.2f}%")
