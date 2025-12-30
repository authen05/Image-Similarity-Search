import streamlit as st
import os
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Visual Search AI",
    page_icon="🔍",
    layout="wide"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        height: 50px;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: white;
        color: #FF4B4B;
        border: 2px solid #FF4B4B;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODEL & FUNCTIONS ---
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

# --- 4. SIDEBAR & SETTINGS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1150/1150626.png", width=80)
    st.header("⚙️ Settings")
    
    # NEW: Slider to control how strict the AI is
    similarity_threshold = st.slider(
        "Match Threshold (%)", 
        min_value=0, 
        max_value=100, 
        value=70, 
        help="If match similarity is below this number, show an error."
    ) / 100.0
    
    st.divider()
    
    image_folder = 'images'
    try:
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        st.success(f"Database: {len(image_files)} images.")
    except FileNotFoundError:
        st.error("Error: 'images' folder missing!")
        image_files = []

# --- 5. MAIN APP ---
st.title("🔍 Image Similarity Search")
st.markdown("---")

model = load_model()

# Indexing (Only runs once)
if 'features_db' not in st.session_state and image_files:
    with st.spinner("Indexing Database..."):
        features_list = []
        for img_file in image_files:
            img = Image.open(os.path.join(image_folder, img_file))
            features = extract_features(img, model)
            features_list.append(features)
        st.session_state['features_db'] = np.array(features_list)
        st.session_state['file_names'] = image_files

# Search Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Upload Photo")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        query_img = Image.open(uploaded_file)
        st.image(query_img, caption="Query Image", use_column_width=True)
        search_click = st.button("🚀 Search Similar Images")
    else:
        search_click = False

with col2:
    st.subheader("2. Results")
    
    if search_click and uploaded_file and 'features_db' in st.session_state:
        with st.spinner("Analyzing..."):
            
            # Extract features
            query_features = extract_features(query_img, model).reshape(1, -1)
            
            # Search logic
            neighbors = NearestNeighbors(n_neighbors=5, metric='cosine')
            neighbors.fit(st.session_state['features_db'])
            distances, indices = neighbors.kneighbors(query_features)
            
            # NEW: Error Handling Logic
            best_score = 1 - distances[0][0] # Score of the very best match
            
            if best_score < similarity_threshold:
                # ERROR CONDITION
                st.error(f"❌ No matching images found!")
                st.warning(f"""
                The closest match was only **{int(best_score*100)}%** similar.
                This is below your threshold of **{int(similarity_threshold*100)}%**.
                
                **Why?**
                - We don't have this object in the database.
                - The angle or lighting is very different.
                """)
            else:
                # SUCCESS CONDITION
                results_cols = st.columns(3)
                count = 0
                for i, idx in enumerate(indices[0]):
                    score = 1 - distances[0][i]
                    
                    # Only show matches that pass the threshold
                    if score >= similarity_threshold:
                        match_file = st.session_state['file_names'][idx]
                        match_path = os.path.join(image_folder, match_file)
                        
                        with results_cols[count % 3]:
                            st.image(match_path, use_column_width=True)
                            st.caption(f"**Match {count+1}**\nSimilarity: {int(score*100)}%")
                        count += 1
