import os
import cv2
import tempfile
import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import gdown
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from facenet_pytorch import MTCNN

st.set_page_config(page_title="Deepfake Detection AI - Multi-Domain", page_icon="🛡️", layout="wide")

st.title("🛡️ Robust Deepfake Detection AI (Spatial, Frequency & TTA)")
st.write("Equipped with **Test-Time Augmentation (TTA)** to handle WhatsApp and camera compression artifacts.")

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
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x)

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
        
        return cam

def compute_fft_spectrum(image_pil):
    gray = np.array(image_pil.convert('L'))
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
    norm_spectrum = cv2.normalize(magnitude_spectrum, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    fft_color = cv2.applyColorMap(norm_spectrum, cv2.COLORMAP_JET)
    return cv2.cvtColor(fft_color, cv2.COLOR_BGR2RGB)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def filter_group_boxes(boxes, probs, img_w, img_h):
    if boxes is None or probs is None:
        return []

    valid = []
    img_area = img_w * img_h

    for box, prob in zip(boxes, probs):
        if prob is not None and prob >= 0.50:
            w = box[2] - box[0]
            h = box[3] - box[1]
            box_area = w * h
            
            # Accepts both small group faces and close-up selfies
            if (box_area / img_area) >= 0.001:
                valid.append((box, prob))

    valid.sort(key=lambda x: x[1], reverse=True)
    
    filtered_boxes = []
    for box, prob in valid:
        keep = True
        for f_box in filtered_boxes:
            if calculate_iou(box, f_box) > 0.30:
                keep = False
                break
        if keep:
            filtered_boxes.append(box)

    return filtered_boxes

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
target_layer = model.backbone.features[-1]
grad_cam = GradCAM(model, target_layer)

mtcnn = MTCNN(image_size=224, margin=20, keep_all=True, post_process=False, device=device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_with_tta(crop_pil, model):
    orig_tensor = transform(crop_pil).unsqueeze(0).to(device)
    
    # Smooth out WhatsApp JPEG compression artifacts
    flipped_pil = crop_pil.transpose(Image.FLIP_LEFT_RIGHT)
    smooth_pil = crop_pil.filter(ImageFilter.GaussianBlur(radius=0.7))
    
    flip_tensor = transform(flipped_pil).unsqueeze(0).to(device)
    smooth_tensor = transform(smooth_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        score_orig = torch.sigmoid(model(orig_tensor)).item()
        score_flip = torch.sigmoid(model(flip_tensor)).item()
        score_smooth = torch.sigmoid(model(smooth_tensor)).item()

    # Averaged score suppresses compression noise spikes
    avg_score = (score_orig * 0.4) + (score_flip * 0.3) + (score_smooth * 0.3)
    return avg_score, orig_tensor

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=400)

    with st.spinner("Analyzing multi-domain features and applying TTA..."):
        try:
            img_w, img_h = img.size
            boxes, probs = mtcnn.detect(img)
            valid_boxes = filter_group_boxes(boxes, probs, img_w, img_h)

            if not valid_boxes:
                st.warning("⚠️ No clear face detected by MTCNN. Running fallback central analysis...")
                crop = img.resize((224, 224))
                prob_real, tensor_in = predict_with_tta(crop, model)
                cam_map = grad_cam.generate_heatmap(tensor_in)
                fft_spectrum = compute_fft_spectrum(crop)
                
                pred_label = "REAL" if prob_real >= 0.38 else "FAKE"
                conf_score = prob_real if pred_label == "REAL" else (1.0 - prob_real)
                
                st.info(f"**Fallback Verdict:** {pred_label} (Confidence: {conf_score * 100:.2f}%)")
            else:
                st.markdown(f"### 👥 Detected {len(valid_boxes)} Face(s)")
                st.markdown("---")

                for idx, box in enumerate(valid_boxes, 1):
                    x1, y1, x2, y2 = [int(b) for b in box]
                    margin = 25
                    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
                    x2, y2 = min(img_w, x2 + margin), min(img_h, y2 + margin)

                    cropped_face = img.crop((x1, y1, x2, y2))
                    
                    prob_real, tensor_in = predict_with_tta(cropped_face, model)
                    cam_map = grad_cam.generate_heatmap(tensor_in)

                    face_cv = np.array(cropped_face.resize((224, 224)))
                    heatmap_resized = cv2.resize(cam_map, (224, 224))
                    heatmap_uint8 = np.uint8(255 * heatmap_resized)
                    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                    overlay = cv2.addWeighted(face_cv, 0.6, heatmap_color, 0.4, 0)

                    fft_spectrum = compute_fft_spectrum(cropped_face)

                    # Calibrated threshold accounting for social media compression
                    if prob_real >= 0.38:
                        pred_label = "REAL"
                        conf_score = prob_real
                    else:
                        pred_label = "FAKE"
                        conf_score = 1.0 - prob_real

                    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                    with col1:
                        st.image(cropped_face, caption=f"Person #{idx} Crop", use_column_width=True)
                    with col2:
                        st.image(overlay, caption=f"Person #{idx} Grad-CAM", use_column_width=True)
                    with col3:
                        st.image(fft_spectrum, caption=f"Person #{idx} 2D-FFT Spectrum", use_column_width=True)
                    with col4:
                        if pred_label == "REAL":
                            st.success(f"**Verdict:** {pred_label}")
                        else:
                            st.error(f"**Verdict:** {pred_label}")
                        st.write(f"• **Real Score:** {prob_real:.4f}")
                        st.write(f"• **Confidence:** {conf_score * 100:.2f}%")

                    st.markdown("---")

        except Exception as e:
            st.error("Error analyzing image. Please try another file.")
