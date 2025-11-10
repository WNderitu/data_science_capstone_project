import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Malaria Parasite (P.vivax) Detector using YOLOv8n",
    layout="wide"
)

# --- CONFIGURATION ---
model_path = "best.onnx" # Path to the ONNX model file
classes_file = "classes.txt"

@st.cache_resource # Cache the model loading
def load_onnx_model(model_path):
    """Loads the ONNX model using OpenCV's DNN module."""
    try:
        net = cv2.dnn.readNet(model_path)
        return net
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}")
        return None

@st.cache_data # Cache class names
def load_class_names(classes_file):
    """Reads class names from a text file."""
    try:
        with open(classes_file, "r") as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        return []

# Load model and classes
net = load_onnx_model(model_path)
class_names = load_class_names(classes_file)

st.title("🔬 Malaria Parasite (P.vivax) Detection using YOLOV8n)")

# Function to process image and run detection
def process_image(net, image, conf_threshold, class_names):
    """Runs detection, draws bounding boxes on the image and returns class counts"""
    
    # YOLOv8 expects a 640x640 input
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640
    img_cv = np.array(image.convert("RGB")) # Prepare image for YOLOv8 (resize and normalize)
    # OpenCV's blobFromImage: resize, normalize, swap R/B, central crop=False
    blob = cv2.dnn.blobFromImage(img_cv, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False) 
    net.setInput(blob)

    # Run inference
    preds = net.forward()
    
    # Process the detection results (YOLOv8 output layer)
    # The output is (1, N, 85) where N is number of boxes, and 85 is [box, objectness, 80 classes]
    detections = preds[0] 

    # Rescale bounding box coordinates
    cols = img_cv.shape[1]
    rows = img_cv.shape[0]
    
    # We'll use a simple post-processing loop for demonstration
    boxes = []
    confidences = []
    class_ids = []

    # Define the numerical IDs that represent a parasite
    # trophozoite (1), ring (2), schizont (3), gametocyte (4), difficult (6)
    parasite_IDs = {1, 2, 3, 4, 6} # Use a set for fast lookup

    # New dictionary to store counts
    class_counts = {name: 0 for name in class_names}

    for i in range(detections.shape[0]):
        row = detections[i]
        confidence = row[4]
        
        if confidence > conf_threshold:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)

            # Ensure class_id is within bounds
            num_classes = len(class_names)
            if class_id >= num_classes:
                 # Skip this detection if the class ID is invalid
                continue
            
            if classes_scores[class_id] > 0.0:
                # Get box coordinates (unchanged)
                center_x = int(row[0] * cols)
                center_y = int(row[1] * rows)
                width = int(row[2] * cols)
                height = int(row[3] * rows)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.45)

    # Safely convert indices to a flat list of integers.
    # This handles cases where NMS returns a 1D, 2D, or even an empty array.
    if len(indices) > 0:
        indices = indices.flatten()
    else:
        # If indices is empty (no boxes passed NMS), skip drawing
        return img_cv, class_counts 
    
    # Draw boxes and update counts
    for i in indices:
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        class_id = class_ids[i]

        # Increment the count for the detected class
        detected_class_name = class_names[class_id]
        class_counts[detected_class_name] += 1

        # Draw rectangle and label
        is_parasite = class_id in parasite_IDs
        label = f"{class_names[class_id]}: {confidences[i]:.2f}"

        if is_parasite:
            # Color for any parasite stage
            color = (255, 0, 0) # Red
        else:
            # Color for non-parasite (e.g., red blood cell,leukocyte)
            color = (0, 255, 0) # Green

        cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img_cv, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return img_cv, class_counts

# Sidebar for user inputs
# --- Sidebar ---
st.sidebar.header("Model Settings")
confidence_threshold = st.sidebar.slider(
    label='Confidence Threshold', 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.05
)

# -- User Interface -- 
st.header("Upload image of blood smear slide")
uploaded_files = st.file_uploader(
    "Choose one or more image files (e.g., JPG, PNG)",
    type=['jpg', 'jpeg', 'png', 'bmp'],
    accept_multiple_files=True # Allow multiple file uploads
)

if uploaded_files and net and class_names:
    st.subheader(f"Processing {len(uploaded_files)} Images...")

    if st.button("Next: Click! To run detection on uploaded images"):
        progress_bar = st.progress(0) # Initialize progress bar
        total_images = len(uploaded_files)
    
        # Loop over all uploaded files
        for i, file in enumerate(uploaded_files):
            # Open image with PIL
            image = Image.open(file)
        
            # Run detection and get BOTH the image and the counts!
            detected_img_cv, class_counts = process_image(net, image, confidence_threshold, class_names)
        
            # Convert OpenCV (BGR) to Streamlit (RGB)
            detected_img_rgb = cv2.cvtColor(detected_img_cv, cv2.COLOR_BGR2RGB)
        
            # Display results in two columns (one for image, one for data)
            col_img, col_data = st.columns([2, 1])

            with col_img:
                st.image(
                    detected_img_rgb, 
                    caption=f"Processed: {file.name}", 
                    use_container_width=True
                )
        
            with col_data:
                st.markdown(f"### Results for **{file.name}**")
                # 1. calculate total parasite count
                # Use the class names corresponding to IDs 1, 2, 3, 4, 6
                # Assuming your class names are: 'trophozoite', 'ring', 'schizont', 'gametocyte', 'difficult'
                parasite_stages = ['trophozoite', 'ring', 'schizont', 'gametocyte', 'difficult']
                total_parasite_count = sum(class_counts.get(stage, 0) for stage in parasite_stages)
                total_detections = sum(class_counts.values())

                # 2. display total parasite count
                st.metric(
                    label=" **Total parasite count (All Stages)**",
                    value=total_parasite_count,
                    delta=f"{total_parasite_count} out of {total_detections} total blood cells detected"
                )
                # 3. Display individual stage counts for a detailed view
                st.markdown("**Detailed Stage Counts:**")
                
                # Loop through the class counts and display
                for class_name, count in class_counts.items():
                    # Display all classes
                    st.metric(label=f"Count of **{class_name.title()}**:", value=count)
                # 4. Display total detections
                st.info(f"**Total Detections:** {total_detections}")

            # Separator for next image
            st.divider() 

            # Update progress bar
            progress_bar.progress((i + 1) / total_images)

        progress_bar.empty()
        st.success("Detection and quantification complete!")
elif not net:
    st.error("ONNX model could not be loaded. Please check the path and file integrity.")
