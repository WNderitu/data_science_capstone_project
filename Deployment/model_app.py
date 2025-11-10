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
model_path = "yolov8.onnx" # Path to the ONNX model file 
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

    # TRANSPOSE THE ARRAY
    # This converts (11, 8400) to (8400, 11) for correct iteration.
    detections = detections.T
    
    # Rescale bounding box coordinates
    cols = img_cv.shape[1]
    rows = img_cv.shape[0]
    
    # post-processing loop to filter boxes based on confidence threshold
    boxes = []
    confidences = []
    class_ids = []

    # --- parasite_IDs and Colour_MAP here ---
    # red blood cell (ID O)
    # trophozoite (ID 1)
    # ring (ID 2)
    # schizont (ID 3)
    # gametocyte (ID 4)
    # leukocyte (ID 5)
    # difficult (ID 6)
    
    # 0: red blood cell
    # 1: trophozoite
    # 2: ring
    # 3: schizont
    # 4: gametocyte
    # 5: leukocyte (if you have it)
    # 6: difficult

    # Define your colors (BGR format for OpenCV)
    COLOR_MAP = {
        'red blood cell': (0, 0, 255), # Red for Red Blood Cell (ID 0)
        'trophozoite': (255, 0, 0),   # Blue for Trophozoite (ID 1)
        'ring': (0, 255, 0),          # Green for Ring (ID 2)
        'schizont': (0, 255, 255),    # Yellow for Schizont (ID 3)
        'gametocyte': (255, 0, 255),  # Magenta for Gametocyte (ID 4)
        'difficult': (0, 165, 255),   # Orange for Difficult (ID 6)
        # Add any other classes here, or define a default for unknown/unspecified
        'leukocyte': (255, 255, 255),  # white for Leukocyte (ID 5)
        'default': (128, 128, 128)     # White for any unspecified class
    }

    # Define the numerical IDs that represent a parasite
    # trophozoite (1), ring (2), schizont (3), gametocyte (4), difficult (6)
    parasite_IDs = {1, 2, 3, 4, 6} 

    # New dictionary to store counts
    class_counts = {name: 0 for name in class_names}

    # List to track max confidence found
    max_conf_found = 0.0

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
                # Scale coordinates from 0-640 range to original pixel dimensions (cols/rows)
                # The YOLOv8 model outputs normalized coordinates relative to the 640x640 input.
                
                # Calculate scaling factors
                x_scale = img_cv.shape[1] / INPUT_WIDTH  # cols / 640
                y_scale = img_cv.shape[0] / INPUT_HEIGHT # rows / 640
                
                # Get box coordinates and scale them
                center_x = row[0] * x_scale
                center_y = row[1] * y_scale
                width = row[2] * x_scale
                height = row[3] * y_scale
                
                # Convert to top-left corner (x, y)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                w = int(width)
                h = int(height)
                
                boxes.append([x, y, w, h]) # Note: Use w and h for clarity
                confidences.append(float(confidence))
                class_ids.append(class_id)

    if len(boxes) == 0:
        # If no boxes passed the confidence threshold, return immediately.
        return img_cv, class_counts
    
    # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.9)

    # Safely convert indices to a flat list of integers.
    # This handles cases where NMS returns a 1D, 2D, or even an empty array.
    if len(indices) > 0:
        indices = indices.flatten()

    if len(indices) == 0:
        return img_cv, class_counts 
    
    # Draw boxes and update counts
    for i in indices:
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        class_id = class_ids[i]

        # Increment the count for the detected class
        detected_class_name = class_names[class_id]
        class_counts[detected_class_name] += 1

        # --- Colour selection logic ---
        # Get color based on the detected class name, with a fallback default color
        color = COLOR_MAP.get(detected_class_name, COLOR_MAP['default'])

        # Draw rectangle and label
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, 2)

        # Check if the detected class is a parasite
        is_parasite = class_id in parasite_IDs
        label = f"{detected_class_name}: {confidences[i]:.2f}"

        # choose font and scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 1

        # Calculate text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Define position for text background rectangle
        text_padding_x = 3 # Horizontal padding
        text_padding_y = 1 # Vertical padding
        
        # Top-left corner of the text background
        text_bg_x1 = x 
        text_bg_y1 = y - text_height - baseline - (text_padding_y * 2) # More precise top calculation
        
        # Bottom-right corner of the text background
        text_bg_x2 = x + text_width + (text_padding_x * 2)
        text_bg_y2 = y # Align bottom of text background with top of bounding box
        
        # Ensure the background rectangle doesn't go off the top of the image
        if text_bg_y1 < 0:
            text_bg_y1 = y + 2 # If it goes too high, place it just below the box
            text_bg_y2 = y + text_height + baseline + (text_padding_y * 2) + 2
            text_y_pos = text_bg_y1 + text_height + baseline - text_padding_y # Text position inside new bottom rect
        else:
            text_y_pos = y - baseline - text_padding_y # Text position inside top rect

        # # Draw a filled rectangle as a background for the text
        # text_bg_color = (0, 0, 0) # Black background
        # cv2.rectangle(img_cv, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), text_bg_color, -1) # -1 for filled
        
        # Determine text color for contrast
        text_color = (0,0,0) # Black text for better visibility
        
        # Put the text on the image
        cv2.putText(img_cv, label, (x + text_padding_x, text_y_pos), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

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
                parasite_stages = ['trophozoite', 'ring', 'schizont', 'gametocyte', 'difficult']
                total_parasite_count = sum(class_counts.get(stage, 0) for stage in parasite_stages)
                total_detections = sum(class_counts.values())

                # 2. Calculate Parasitemia Rate
                # Parasitemia = (Total Parasites / Total Detections) * 100
                if total_detections > 0:
                    parasitemia = (total_parasite_count / total_detections) * 100
                    parasitemia_display = f"{parasitemia:.2f} %"
                else:
                    parasitemia = 0.0
                    parasitemia_display = "0.00 %"

                # 3. display total parasite count
                st.metric(
                    label=" **Total parasite count (All Stages)**",
                    value=total_parasite_count,
                    delta=f"{total_parasite_count} out of {total_detections} total blood cells detected"
                )

                # 4. Display Parasitemia Rate
                st.metric(
                    label=" **Estimated Parasitemia Rate**",
                    value=parasitemia_display,
                    help="Calculated as: (Total Parasite Detections / Total Cell Detections) * 100"
                )

                # 5. Display individual stage counts for a detailed view
                st.markdown("**Detailed Stage Counts:**")
                
                # Loop through the class counts and display them
                for class_name, count in class_counts.items():
                    # Display all classes
                    st.metric(label=f"Count of **{class_name.title()}**:", value=count)

                st.markdown("**Detailed Parasite Stage Counts:**")
                
                # Loop through the class counts, but display ONLY the parasite stages
                # filtered_display = False
                # for class_name in parasite_stages:
                #     count = class_counts.get(class_name, 0)
                #     if count > 0:
                #         st.metric(label=f"Count of **{class_name.title()}**:", value=count)
                #         filtered_display = True

                # # Message if no specific stages were found (only non-parasites)
                # if not filtered_display and total_parasite_count > 0:
                #      st.warning("All parasites detected are of an unspecified stage.")
                # elif total_parasite_count == 0:
                #      st.success("No parasite stages detected in this image.")

                # 5 Display total detections
                st.info(f"**Total Objects Counted (Parasites + Non-Parasites):** {total_detections}")

            # Separator for next image
            st.divider() 

            # Update progress bar
            progress_bar.progress((i + 1) / total_images)

        progress_bar.empty()
        st.success("Detection and quantification complete!")
elif not net:
    st.error("ONNX model could not be loaded. Please check the path and file integrity.")
