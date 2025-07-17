import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set page config
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="wide"
)

# App title and description
st.title("YOLO Object Detection")
st.markdown("Upload an image to detect objects using YOLOv8")

# API endpoint
API_URL = "http://0.0.0.0:8000/predict"

def draw_boxes(image, predictions):
    """Draw bounding boxes on the image based on predictions"""
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(np.array(image))
    
    # Generate random colors for different classes
    colors = plt.cm.hsv(np.linspace(0, 1, 20))
    
    # Draw each bounding box
    for pred in predictions:
        # Get coordinates
        x1 = pred["bbox"]["x1"]
        y1 = pred["bbox"]["y1"]
        x2 = pred["bbox"]["x2"]
        y2 = pred["bbox"]["y2"]
        
        # Calculate width and height
        width = x2 - x1
        height = y2 - y1
        
        # Get class and confidence
        class_name = pred["class_name"]
        confidence = pred["confidence"]
        
        # Get color based on class_id
        color = colors[pred["class_id"] % len(colors)]
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1), width, height, 
            linewidth=2, 
            edgecolor=color, 
            facecolor='none'
        )
        
        # Add patch to plot
        ax.add_patch(rect)
        
        # Add label
        plt.text(
            x1, y1-5, 
            f"{class_name}: {confidence:.2f}", 
            color=color, 
            fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.7)
        )
    
    plt.axis('off')
    return fig

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    st.subheader("Original Image")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Create two columns for results
    col1, col2 = st.columns(2)
    
    with col1:
        # Button to trigger prediction
        if st.button("Detect Objects"):
            with st.spinner("Detecting objects..."):
                try:
                    # Prepare the file for the API request
                    files = {"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
                    
                    # Make API request
                    response = requests.post(API_URL, files=files)
                    
                    # Check if request was successful
                    if response.status_code == 200:
                        # Get prediction results
                        result = response.json()
                        predictions = result["predictions"]
                        
                        # Display number of objects detected
                        st.success(f"Detected {len(predictions)} objects!")
                        
                        # Display image with bounding boxes
                        if predictions:
                            st.subheader("Detection Results")
                            fig = draw_boxes(image, predictions)
                            st.pyplot(fig)
                        else:
                            st.info("No objects detected in the image.")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    
    with col2:
        # Display prediction details
        if "predictions" in locals():
            st.subheader("Detection Details")
            
            if predictions:
                # Create a table of predictions
                prediction_data = []
                for i, pred in enumerate(predictions):
                    prediction_data.append({
                        "Object": i+1,
                        "Class": pred["class_name"],
                        "Confidence": f"{pred['confidence']:.2f}",
                        "Position": f"({int(pred['bbox']['x1'])}, {int(pred['bbox']['y1'])}) to ({int(pred['bbox']['x2'])}, {int(pred['bbox']['y2'])})"
                    })
                
                st.table(prediction_data)
            else:
                st.info("No objects detected in the image.")

# Add information about the model
st.sidebar.title("About")
st.sidebar.info(
    "This application uses YOLOv8 for object detection. "
    "Upload an image to detect objects and their locations."
)

# Add model information
try:
    model_info_response = requests.get("http://0.0.0.0:8000/model/info")
    if model_info_response.status_code == 200:
        model_info = model_info_response.json()
        st.sidebar.subheader("Model Information")
        st.sidebar.text(f"Model Path: {model_info['model_path']}")
        st.sidebar.text(f"Confidence Threshold: {model_info['confidence_threshold']}")
        st.sidebar.text(f"Model Loaded: {model_info['model_loaded']}")
except:
    st.sidebar.warning("Could not fetch model information.")