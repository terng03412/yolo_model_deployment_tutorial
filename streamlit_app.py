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
    # Get original image dimensions
    original_width, original_height = image.size
    
    # Create figure with proper aspect ratio
    aspect_ratio = original_width / original_height
    if aspect_ratio > 1:
        fig_width = 12
        fig_height = 12 / aspect_ratio
    else:
        fig_height = 12
        fig_width = 12 * aspect_ratio
    
    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
    ax.imshow(np.array(image))
    
    # Generate random colors for different classes
    colors = plt.cm.hsv(np.linspace(0, 1, 20))
    
    # Draw each bounding box
    for pred in predictions:
        # Get coordinates (these should already be in the correct scale from the API)
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
            linewidth=3, 
            edgecolor=color, 
            facecolor='none'
        )
        
        # Add patch to plot
        ax.add_patch(rect)
        
        # Add label with better positioning
        label_y = max(y1 - 10, 10)  # Ensure label doesn't go off-screen
        plt.text(
            x1, label_y, 
            f"{class_name}: {confidence:.2f}", 
            color='white',
            fontsize=12, 
            fontweight='bold',
            bbox=dict(facecolor=color, alpha=0.8, pad=2)
        )
    
    # Set axis limits to match image dimensions exactly
    ax.set_xlim(0, original_width)
    ax.set_ylim(original_height, 0)  # Invert y-axis for image coordinates
    ax.set_aspect('equal')
    plt.axis('off')
    plt.tight_layout()
    
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
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    
                    # Get the original filename and content type
                    filename = uploaded_file.name
                    content_type = uploaded_file.type
                    
                    # Prepare the file for the API request
                    files = {"file": (filename, uploaded_file.getvalue(), content_type)}
                    
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