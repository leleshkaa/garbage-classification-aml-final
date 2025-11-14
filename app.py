import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import torch.nn.functional as F

st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("Garbage Classifier")

class GarbageCNN(nn.Module):
    def __init__(self, num_classes):
        super(GarbageCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_local_model():
    device = torch.device("cpu")
    class_file = "class_names.txt"
    model_file = "best_garbage_model.pth"
    class_names = None
    if os.path.isfile(class_file):
        with open(class_file, "r", encoding="utf-8") as f:
            class_names = [c.strip() for c in f.readlines() if c.strip()]
    elif os.path.isdir("classes"):
        class_names = sorted([d for d in os.listdir("classes") if os.path.isdir(os.path.join("classes", d))])
    else:
        class_names = []
    model = None
    if os.path.isfile(model_file) and len(class_names)>0:
        model = GarbageCNN(num_classes=len(class_names))
        state = torch.load(model_file, map_location=device)
        try:
            model.load_state_dict(state)
        except:
            from collections import OrderedDict
            new_state = OrderedDict()
            for k, v in state.items():
                new_state[k.replace("module.", "")] = v
            model.load_state_dict(new_state)
        model.to(device).eval()
    return model, class_names

model, class_names = load_local_model()

uploaded_img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if uploaded_img is not None:
    try:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, use_column_width=True)
        if model is None or len(class_names)==0:
            st.error("Model or class_names.txt not found in app folder.")
        else:
            transform = transforms.Compose([
                transforms.Resize((128,128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(tensor)
                probs = F.softmax(outputs, dim=1).cpu().numpy().ravel()
            idx = probs.argmax()
            st.write(f"{class_names[idx]} — {probs[idx]*100:.2f}%")
    except Exception as e:
        st.error(str(e))
