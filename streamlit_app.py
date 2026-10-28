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

st.set_page_config(page_title="Deepfake Detection AI with Grad-CAM", page_icon="🛡️", layout="centered")

st.title("🛡️ Deepfake Detection AI + Grad-CAM")
st.write("Upload an image to evaluate facial authenticity and view the Grad-CAM visual explainability heatmap.")

device = torch.device("cpu")
CHECKPOINT_PATH = "best_deepfake_model.pt"
DRIVE_FILE_ID = "1IKlstbHz9bq7sVT_1ULIjuQGk6HKFhVS"

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

# Grad-CAM Engine using PyTorch Backward Hooks
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        logit = output.squeeze()
        logit.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.clamp(cam, min=0)
        
        cam = cam.cpu().detach().numpy()
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        
        return cam, torch.sigmoid(logit).item()

@st.cache_resource
def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        with st.spinner("Downloading model weights from Google Drive..."):
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, CHECKPOINT_PATH, quiet=False)

    model = DeepfakeDetector().to(device)
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

model = load_model()

# Extract target layer for Grad-CAM
target_layer = model.backbone.features[-1]
grad_cam = GradCAM(model, target_layer)

mtcnn = MTCNN(image_size=224, margin=30, keep_all=True, post_process=False, device=device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Source Image", use_column_width=True)

    with st.spinner("Detecting faces & generating Grad-CAM heatmaps..."):
        try:
            boxes, probs = mtcnn.detect(img)
            valid_boxes = [box for box, prob in zip(boxes, probs) if prob is not None and prob >= 0.70] if boxes is not None else []

            if not valid_boxes:
                st.warning("⚠️ No clear human faces detected.")
            else:
                st.markdown(f"### 👥 Detected {len(valid_boxes)} Face(s)")
                st.markdown("---")

                img_w, img_h = img.size
                
                for idx, box in enumerate(valid_boxes, 1):
                    x1, y1, x2, y2 = [int(b) for b in box]
                    margin = 20
                    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
                    x2, y2 = min(img_w, x2 + margin), min(img_h, y2 + margin)

                    cropped_face = img.crop((x1, y1, x2, y2))
                    tensor_in = transform(cropped_face).unsqueeze(0).to(device)

                    cam_map, prob_real = grad_cam.generate_heatmap(tensor_in)

                    face_cv = np.array(cropped_face.resize((224, 224)))
                    heatmap_resized = cv2.resize(cam_map, (224, 224))
                    heatmap_uint8 = np.uint8(255 * heatmap_resized)
                    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

                    overlay = cv2.addWeighted(face_cv, 0.6, heatmap_color, 0.4, 0)

                    pred_label = "REAL" if prob_real >= 0.50 else "FAKE"
                    conf_score = prob_real if pred_label == "REAL" else (1.0 - prob_real)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(cropped_face, caption=f"Person #{idx} Crop", use_column_width=True)
                    with col2:
                        st.image(overlay, caption=f"Person #{idx} Grad-CAM", use_column_width=True)
                    with col3:
                        if pred_label == "REAL":
                            st.success(f"**Verdict:** {pred_label}")
                        else:
                            st.error(f"**Verdict:** {pred_label}")
                        st.write(f"• **Confidence:** {conf_score * 100:.2f}%")
                        st.write(f"• **Real Score:** {prob_real:.4f}")

                    st.markdown("---")

        except Exception as e:
            st.error("Error analyzing image with Grad-CAM.")
