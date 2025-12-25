import streamlit as st
import os
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="AI Image Search", layout="wide")

st.title("🧠 Deep Learning Image Similarity Search")
st.markdown("This system uses **ResNet50** to extract visual features and find similar images in your database.")

# --- 2. BACKEND LOGIC (The "Brain") ---

# A. Load the Pre-trained Model (Cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    # Load ResNet50 pre-trained on ImageNet
    model = models.resnet50(pretrained=True)
    # Remove the last layer (classification) because we only want features
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval() # Set to evaluation mode
    return feature_extractor

# B. Image Preprocessing (Resize & Normalize)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# C. Function to Turn an Image into a Vector
def extract_features(image, model):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0) # Add batch dimension
    
    with torch.no_grad():
        features = model(img_tensor)
    
    # Flatten the result to a 1D vector (size 2048)
    return features.flatten().numpy()

# --- 3. LOADING THE DATABASE ---
model = load_model()

# Load all images from the 'images' folder
image_folder = 'images'
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# If no images found, show a warning
if not image_files:
    st.error(f"No images found in the '{image_folder}' folder! Please upload some images to GitHub.")
else:
    # Compute features for all database images (One-time setup)
    # Ideally, we would save these to a file, but for a demo, we compute them live.
    
    # Check if we already have features in session state to save time
    if 'features_db' not in st.session_state:
        with st.spinner(f"Indexing {len(image_files)} images from database..."):
            features_list = []
            for img_file in image_files:
                img_path = os.path.join(image_folder, img_file)
                img = Image.open(img_path)
                features = extract_features(img, model)
                features_list.append(features)
            
            st.session_state['features_db'] = np.array(features_list)
            st.session_state['file_names'] = image_files
            st.success("Database Indexed Successfully!")

    # --- 4. USER INTERFACE ---
    uploaded_file = st.file_uploader("Upload a Query Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            query_img = Image.open(uploaded_file)
            st.image(query_img, caption="Your Input", width=300)
            
        with col2:
            st.subheader("🔍 Recommended Matches")
            
            # Extract features for the uploaded image
            query_features = extract_features(query_img, model).reshape(1, -1)
            
            # Find the nearest neighbors
            database_features = st.session_state['features_db']
            
            # We use NearestNeighbors for similarity search
            neighbors = NearestNeighbors(n_neighbors=5, metric='cosine')
            neighbors.fit(database_features)
            distances, indices = neighbors.kneighbors(query_features)
            
            # Display Results
            cols = st.columns(3)
            for i, idx in enumerate(indices[0]):
                # Logic to display results in a grid
                with cols[i % 3]:
                    match_file = st.session_state['file_names'][idx]
                    match_path = os.path.join(image_folder, match_file)
                    match_img = Image.open(match_path)
                    
                    # Calculate similarity score (1 - distance for cosine)
                    score = 1 - distances[0][i]
                    st.image(match_img, caption=f"Match {i+1} ({score:.2f})")
