import streamlit as st
import os
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# --- PAGE CONFIG ---
st.set_page_config(page_title="Visual Search", layout="wide")

# --- DEBUG TITLE (To prove update worked) ---
st.title("✅ UPDATE SUCCESSFUL: New Version 2.0")
st.success("If you see this green box, the code has updated!")

# --- CSS STYLES ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        height: 50px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    return feature_extractor

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image, model):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor)
    return features.flatten().numpy()

# --- SIDEBAR & SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings")
    # THE SLIDER YOU ASKED FOR
    similarity_threshold = st.slider("Strictness (%)", 0, 100, 60, 5) / 100.0
    
    st.divider()
    
    # LOAD IMAGES
    image_folder = 'images'
    if os.path.exists(image_folder):
        # SORTED FIX
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        st.write(f"images loaded: {len(image_files)}")
    else:
        st.error("Images folder missing!")
        image_files = []

# --- MAIN APP ---
model = load_model()

# Index Database
if 'features_db' not in st.session_state and image_files:
    features_list = []
    for img_file in image_files:
        img = Image.open(os.path.join(image_folder, img_file))
        features = extract_features(img, model)
        features_list.append(features)
    st.session_state['features_db'] = np.array(features_list)
    st.session_state['file_names'] = image_files

# UI Layout
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        query_img = Image.open(uploaded_file)
        st.image(query_img, width=250)
        search = st.button("Search")
    else:
        search = False

with col2:
    if search and uploaded_file:
        # Search
        query_features = extract_features(query_img, model).reshape(1, -1)
        neighbors = NearestNeighbors(n_neighbors=5, metric='cosine')
        neighbors.fit(st.session_state['features_db'])
        distances, indices = neighbors.kneighbors(query_features)
        
        # ERROR LOGIC
        best_score = 1 - distances[0][0]
        st.write(f"**Best Match Confidence:** {int(best_score * 100)}%")
        
        if best_score < similarity_threshold:
            st.error("❌ No match found! (Below threshold)")
            st.info(f"Try lowering the strictness slider in the sidebar (Current: {int(similarity_threshold*100)}%)")
        else:
            cols = st.columns(3)
            for i, idx in enumerate(indices[0]):
                score = 1 - distances[0][i]
                if score >= similarity_threshold:
                    match_file = st.session_state['file_names'][idx]
                    st.image(os.path.join(image_folder, match_file), caption=f"{int(score*100)}% match")
