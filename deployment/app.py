import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
import io
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="Malaria Parasite (P.vivax) Detector using YOLOv8n",
    layout="wide"
)

# --- File Paths ---
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'best.onnx')
classes_path = os.path.join(base_path, 'classes.txt')

# --- Validate Files ---
# Check if files exist
if not os.path.exists(model_path):
    st.error(f"ONNX model not found at: {model_path}")
else:
    st.success("ONNX model loaded successfully.")

if not os.path.exists(classes_path):
    st.error(f"Class names file not found at: {classes_path}")
else:
    st.success("Class names file loaded successfully.")

# --- Load Model & Classes ---
@st.cache_resource
def load_onnx_model(model_path):
    try:
        net = cv2.dnn.readNet(model_path)
        return net
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}. Attempted path: {model_path}")
        return None

@st.cache_data
def load_class_names(classes_path):
    try:
        with open(classes_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        st.error(f"Error loading class names: {e}. Attempted path: {classes_path}")
        return []

net = load_onnx_model(model_path)
class_names = load_class_names(classes_path)

st.title("🔬 Malaria Parasite (P.vivax) Detection using YOLOV8n")

# --- Dynamic Sidebar ---
st.sidebar.header("⚙️ Model & Visualization Settings")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
nms_threshold = st.sidebar.slider("NMS Threshold", 0.0, 1.0, 0.35, 0.05)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
show_labels = st.sidebar.checkbox("Show Class Labels", value=True)
show_only_parasites = st.sidebar.checkbox("Show Only Parasite Detections", value=False)
color_scheme = st.sidebar.selectbox("Color Scheme", ["Default", "High Contrast", "Pastel"], index=0)

# --- Image Processing Function ---
def process_image(net, image, conf_threshold, nms_threshold, class_names,
                  show_boxes=True, show_labels=True, show_only_parasites=False, color_scheme="Default"):
    INPUT_WIDTH, INPUT_HEIGHT = 640, 640
    img_cv = np.array(image.convert("RGB"))
    blob = cv2.dnn.blobFromImage(img_cv, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0].T

    boxes, confidences, class_ids = [], [], []
    parasite_IDs = {1, 2, 3, 4, 6}

    # Color maps
    DEFAULT_COLOR_MAP = {
        'red blood cell': (0, 0, 255),
        'trophozoite': (255, 0, 0),
        'ring': (0, 255, 0),
        'schizont': (0, 255, 255),
        'gametocyte': (255, 0, 255),
        'difficult': (0, 165, 255),
        'leukocyte': (255, 255, 255),
        'default': (128, 128, 128)
    }
    HIGH_CONTRAST_MAP = {k: (255, 255, 0) for k in DEFAULT_COLOR_MAP}
    PASTEL_MAP = {k: (200, 180, 255) for k in DEFAULT_COLOR_MAP}

    if color_scheme == "High Contrast":
        COLOR_MAP = HIGH_CONTRAST_MAP
    elif color_scheme == "Pastel":
        COLOR_MAP = PASTEL_MAP
    else:
        COLOR_MAP = DEFAULT_COLOR_MAP

    class_counts = {name: 0 for name in class_names}

    for row in detections:
        confidence = row[4]
        if confidence > conf_threshold:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
            if class_id >= len(class_names):
                continue
            if classes_scores[class_id] > 0.0:
                x_scale = img_cv.shape[1] / INPUT_WIDTH
                y_scale = img_cv.shape[0] / INPUT_HEIGHT
                center_x, center_y, width, height = row[0]*x_scale, row[1]*y_scale, row[2]*x_scale, row[3]*y_scale
                x, y, w, h = int(center_x - width/2), int(center_y - height/2), int(width), int(height)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    if not boxes:
        return img_cv, class_counts

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
    else:
        return img_cv, class_counts

    if show_only_parasites:
        indices = [i for i in indices if class_ids[i] in parasite_IDs]

    for i in indices:
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        detected_class_name = class_names[class_id]
        class_counts[detected_class_name] += 1

        if show_boxes:
            color = COLOR_MAP.get(detected_class_name, COLOR_MAP['default'])
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), color, 2)

        if show_labels:
            label = f"{detected_class_name}: {confidences[i]:.2f}"
            cv2.putText(img_cv, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

    return img_cv, class_counts

# --- User Interface ---
st.header(" 🩸 Upload image of blood smear slide")
uploaded_files = st.file_uploader("Choose one or more image files", type=['jpg','jpeg','png','bmp'], accept_multiple_files=True)

st.sidebar.header("📊 Chart Settings")
chart_mode = st.sidebar.radio(
    'Chart Mode',
    ['Counts', 'Percentages'],
    index=0,
    key='chart_mode_radio'  # unique key to prevent duplicates
)

if uploaded_files and net and class_names:
    st.subheader(f"Processing {len(uploaded_files)} Images...")
    if st.button("  ▶️  Run detection"):
        progress_bar = st.progress(0)
        total_images = len(uploaded_files)
        results_summary = []  # Collect results for CSV export

        for i, file in enumerate(uploaded_files):
            image = Image.open(file)
            detected_img_cv, class_counts = process_image(
                net, image, confidence_threshold, nms_threshold, class_names,
                show_boxes, show_labels, show_only_parasites, color_scheme
            )
            detected_img_rgb = cv2.cvtColor(detected_img_cv, cv2.COLOR_BGR2RGB)
            
            col_img, col_data = st.columns([2,1])
            with col_img:
                st.image(detected_img_rgb, caption=f"Processed: {file.name}", use_container_width=True)
            
            with col_data:
                st.markdown(f"### 🧪 Results for **{file.name}**")
                
                parasite_stages = ['trophozoite','ring','schizont','gametocyte','difficult']
                total_parasite_count = sum(class_counts.get(stage,0) for stage in parasite_stages)
                total_detections = sum(class_counts.values())
                parasitemia = (total_parasite_count/total_detections)*100 if total_detections>0 else 0.0
                parasitemia_display = f"{parasitemia:.2f} %"
                
                st.metric("**Total Parasite Count (All Stages)**", total_parasite_count)
                st.metric(
                    label='**Estimated Parasitemia Rate**',
                    value=parasitemia_display,
                    help=("Calculated as: (Total Parasite Detections / Total Cell Detections) * 100. It estimates the proportion of infected cells among all detected cells.")
                    )
                st.info(f"**Total Objects Counted:** {total_detections}")

                # Class Count Overview
                st.markdown("### 🧫 Class Counts Overview")
                # Create 3 columns for a denser overview
                cols = st.columns(3)
                items = list(class_counts.items())

                # Iterate over the list in steps of 3
                for idx in range(0, len(items), 3):
                    # Column 1 (always exists)
                    class_name_c1, count_c1 = items[idx]
                    cols[0].markdown(f"**{class_name_c1.title()}:** {count_c1}")

                    # Column 2 (if exists)
                    if idx + 1 < len(items):
                        class_name_c2, count_c2 = items[idx+1]
                        cols[1].markdown(f"**{class_name_c2.title()}:** {count_c2}")
    
                    # Column 3 (if exists)
                    if idx + 2 < len(items):
                        class_name_c3, count_c3 = items[idx+2]
                        cols[2].markdown(f"**{class_name_c3.title()}:** {count_c3}")
                        
                # Bar Chart
                counts_df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
                if not counts_df.empty:
                    # Calculate percentages
                    total = counts_df["Count"].sum()
                    counts_df["Percentage"] = (counts_df["Count"] / total) * 100 if total > 0 else 0
                                                          
                    # Choose which column to plot based on mode
                    if chart_mode == 'Counts':
                        x_field = "Count"
                        x_title = "Number of Detections"
                        chart_title = "Detection Counts per Class"
                    else:
                        x_field = "Percentage"
                        x_title = "Detections (%)"
                        chart_title = "Detection Percentage per Class"
                    
                    # Build chart
                    chart = (
                        alt.Chart(counts_df)
                        .mark_bar()
                        .encode(
                            x=alt.X(f"{x_field}:Q", title=x_title),
                            y=alt.Y("Class:N", sort='-x', title="Class Name"),
                            color=alt.Color("Class:N",legend=None),
                            tooltip=[
                                alt.Tooltip('class:N',title="Class"),
                                alt.Tooltip("Count:Q", title="Count"),
                                alt.Tooltip("Percentage:Q", title="Percentage", format=".2f")
                            ]
                        )
                        .properties(width="container",height=200,title=chart_title)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("No detections found to visualize.")
                    
                # Append results for CSV
                results_summary.append({
                    "Image": file.name,
                    "Total Parasites": total_parasite_count,
                    "Total Detections": total_detections,
                    "Parasitemia (%)": f"{parasitemia:.2f}",
                    **{f"Count_{cls}": count for cls, count in class_counts.items()}
                })

            st.divider()
            st.subheader("🧫 Class Counts Overview")
            cols = st.columns(7) # Use 7 columns for 7 classes for maximum spread

            # Use st.metric for the clean box look
            for idx, (class_name, count) in enumerate(class_counts.items()):
                with cols[idx]:
                    st.metric(label=class_name.title(), value=count)
            progress_bar.progress((i+1)/total_images)

        progress_bar.empty()
        st.success("Detection and quantification complete!")

        # --- CSV Export ---
        if results_summary:
            df_results = pd.DataFrame(results_summary)
            csv_buffer = io.StringIO()
            df_results.to_csv(csv_buffer, index=False)
            st.sidebar.download_button(
                label="📥 Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name="malaria_detection_results.csv",
                mime="text/csv",
                help="Export per-image counts and parasitemia rates."
            )
elif not net:
    st.error(" ❌ ONNX model could not be loaded. Please check the path and file integrity.")
