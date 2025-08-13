#!/usr/bin/env python
# coding: utf-8

# # STREAMLIT APP CODE

# In[ ]:


import streamlit as st
import torch
import timm
import numpy as np
import os
from PIL import Image
from datetime import datetime
from torchvision import transforms
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import pandas as pd
import base64
import cv2

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Eye Diagnosis - Anglia Ruskin University", layout="wide")

# ------------------ APPLY BACKGROUND AND STYLE ------------------
def apply_background(image_path, opacity=0.25):
    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url('data:image/png;base64,{encoded}');
            background-size: cover;
            background-attachment: fixed;
        }}

        main label, main p, main span, main div[data-testid="stMarkdownContainer"],
        .overlay, .overlay p, .overlay h1 {{
            color: white !important;
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: white !important;
        }}

        section[data-testid="stSidebar"] {{
            background-color: rgba(0, 0, 0, 0.7) !important;
        }}
        section[data-testid="stSidebar"] * {{
            color: white !important;
        }}

        label {{
            color: white !important;
        }}

        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"],
        .stRadio > div > div {{
            background-color: #222 !important;
            color: white !important;
        }}

        .stButton > button {{
            background-color: #0066cc;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 6px;
        }}

        form button[type="submit"] {{
            background-color: #28a745 !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 6px;
            padding: 10px 20px;
        }}

        .stDataFrame {{
            background-color: rgba(0,0,0,0.4);
            color: white;
        }}
        </style>
    """, unsafe_allow_html=True)

# ------------------ MODEL CONFIG ------------------
data_dir = "D:/AUGMENTED_SPLIT_DATASET"
model_path_b3 = f"{data_dir}/models/EfficientNet-B3_best.pth"
model_path_b4 = f"{data_dir}/models/EfficientNet-B4_best.pth"
class_names = ['Age-related Macular Degeneration', 'Cataract', 'Central Serous Chorioretinopathy [Color Fundus]',
               'Diabetic Retinopathy', 'Disc Edema', 'Glaucoma', 'Healthy', 'Hypertension', 'Macular Scar',
               'Pathological Myopia', 'Retinal Detachment', 'Retinitis Pigmentosa']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model(name, path):
    model = timm.create_model(name, pretrained=False, num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

model_b3 = load_model("tf_efficientnet_b3", model_path_b3)
model_b4 = load_model("tf_efficientnet_b4", model_path_b4)

# ------------------ LOGIN ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    apply_background("assets/HOME_darkened.jpg", 0.25)
    st.title("Staff Login - Eye Diagnosis System")
    access_key = st.text_input("Enter Access Key", type="password")
    if st.button("Login"):
        if access_key == "ai123":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid Access Key")
    st.stop()

# ------------------ NAVIGATION ------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Diagnosis", "History"])
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ------------------ PREDICTION ------------------
def predict(image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out1 = torch.softmax(model_b3(tensor), dim=1)
        out2 = torch.softmax(model_b4(tensor), dim=1)
        final = (out1 + out2) / 2
    probs = final.cpu().numpy().flatten()
    return class_names[np.argmax(probs)], probs, np.argmax(probs)

# ------------------ CORRECTED GRADCAM FUNCTION ------------------
def generate_gradcam(image, model, class_idx):
    model.eval()
    tensor = transform(image).unsqueeze(0).to(device)
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fw = model.blocks[-1].register_forward_hook(forward_hook)
    handle_bw = model.blocks[-1].register_backward_hook(backward_hook)

    output = model(tensor)
    model.zero_grad()
    output[0, class_idx].backward()

    fmap = features[0][0].detach().cpu().numpy()
    grad = gradients[0][0].detach().cpu().numpy()

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img = np.array(image.resize((224, 224)))
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    handle_fw.remove()
    handle_bw.remove()
    return overlay

# ------------------ UPDATED PDF REPORT ------------------
def generate_pdf(name, gender, age, prediction, probs, fundus_image, gradcam_image, filename="report.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, "Eye Diagnosis Report - Anglia Ruskin University")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 90, f"Name: {name}")
    c.drawString(50, height - 110, f"Age: {age}")
    c.drawString(50, height - 130, f"Gender: {gender}")
    c.drawString(50, height - 150, f"Prediction: {prediction}")
    c.drawString(50, height - 170, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = height - 200
    for cls, p in zip(class_names, probs):
        c.drawString(60, y, f"- {cls}: {p:.2f}")
        y -= 15

    fundus_path = "temp_fundus.jpg"
    gradcam_path = "temp_gradcam.jpg"

    fundus_image.save(fundus_path)
    gradcam_image.save(gradcam_path)

    c.drawImage(fundus_path, 50, y - 220, width=220, height=220)
    c.drawImage(gradcam_path, 320, y - 220, width=220, height=220)

    c.save()

    os.remove(fundus_path)
    os.remove(gradcam_path)

    return filename

# ------------------ HOME PAGE ------------------
if page == "Home":
    apply_background("assets/HOME_darkened.jpg", 0.25)
    st.markdown("""
    <div class='overlay'>
    <h1>Eye Disease Diagnosis Platform</h1>
    <p>This AI-powered tool by Anglia Ruskin University allows staff to upload fundus images for diagnosis.
    It provides predictions with Grad-CAM visualization and downloadable reports.</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------ DIAGNOSIS PAGE ------------------
elif page == "Diagnosis":
    apply_background("assets/PREDICT_darkened.png", 0.25)
    st.markdown("<div class='overlay'>", unsafe_allow_html=True)
    st.title("Upload Fundus Image for Diagnosis")
    uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    with st.form("form"):
        name = st.text_input("Patient Name")
        age = st.number_input("Age", 1, 120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        submitted = st.form_submit_button("Diagnose")

    if uploaded_files and submitted:
        for file in uploaded_files:
            image = Image.open(file).convert("RGB")
            st.image(image, caption=file.name, use_column_width=True)
            pred, probs, idx = predict(image)
            st.success(f"Prediction: {pred}")
            df = pd.DataFrame({"Class": class_names, "Probability": probs}).sort_values("Probability", ascending=False)
            st.bar_chart(df.set_index("Class"))
            grad_img = generate_gradcam(image, model_b4, idx)
            st.image(grad_img, caption="Grad-CAM Visualization")

            row = pd.DataFrame([[name, age, gender, pred, file.name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')]],
                               columns=["Name", "Age", "Gender", "Prediction", "Image", "Date"])
            if not os.path.exists("prediction_history.csv"):
                row.to_csv("prediction_history.csv", index=False)
            else:
                row.to_csv("prediction_history.csv", mode='a', header=False, index=False)

            pdf = generate_pdf(name, gender, age, pred, probs, image, Image.fromarray(grad_img), filename=f"Report_{name}_{file.name}.pdf")
            with open(pdf, "rb") as f:
                st.download_button("Download PDF Report", f, file_name=os.path.basename(pdf))
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ HISTORY PAGE ------------------
elif page == "History":
    apply_background("assets/HOME_darkened.jpg", 0.25)
    st.markdown("<div class='overlay'>", unsafe_allow_html=True)
    st.title("Previous Predictions")
    try:
        df = pd.read_csv("prediction_history.csv")
        search_name = st.text_input("Search by Patient Name")
        if search_name:
            df = df[df["Name"].str.contains(search_name, case=False)]

        st.dataframe(df, use_container_width=True)

        edit_id = st.number_input("Enter row index to edit", min_value=0, max_value=len(df)-1, step=1)
        password = st.text_input("Enter edit password", type="password")
        if password == "edit123":
            st.write("Editing row:")
            df.at[edit_id, "Name"] = st.text_input("Edit Name", value=df.at[edit_id, "Name"])
            df.at[edit_id, "Age"] = st.number_input("Edit Age", value=int(df.at[edit_id, "Age"]))
            df.at[edit_id, "Gender"] = st.selectbox("Edit Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(df.at[edit_id, "Gender"]))
            df.at[edit_id, "Prediction"] = st.text_input("Edit Prediction", value=df.at[edit_id, "Prediction"])
            if st.button("Save Changes"):
                df.to_csv("prediction_history.csv", index=False)
                st.success("Changes saved successfully.")

        del_index = st.number_input("Enter row index to delete", min_value=0, max_value=len(df)-1, step=1, key="del")
        if st.button("Delete Entry"):
            df = df.drop(index=del_index).reset_index(drop=True)
            df.to_csv("prediction_history.csv", index=False)
            st.success("Entry deleted.")

        csv = df.to_csv(index=False).encode()
        st.download_button("Download History CSV", csv, file_name="Prediction_History.csv")

    except:
        st.warning("No history found.")
    st.markdown("</div>", unsafe_allow_html=True)


# In[ ]:




