import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
from io import BytesIO
from ultralytics import YOLO
from paddleocr import PaddleOCR
import pandas as pd
import os
from typing import Tuple, List, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages the loading and caching of detection and OCR models."""
    
    @st.cache_resource
    def load_yolo_model(model_path: str) -> YOLO:
        """Load and cache YOLO model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        return YOLO(model_path)

    @st.cache_resource
    def load_ocr_model() -> PaddleOCR:
        """Load and cache PaddleOCR model."""
        try:
            return PaddleOCR(
                rec_model_dir="paddle",
                use_angle_cls=True,
                lang='en',
                show_log=False
            )
        except Exception as e:
            logger.error(f"Error loading OCR model: {e}")
            raise

class ImageProcessor:
    """Handles image processing operations including detection and OCR."""
    
    def __init__(self, yolo_model: YOLO, ocr_model: PaddleOCR):
        self.yolo_model = yolo_model
        self.ocr_model = ocr_model

    def detect_objects(self, image: np.ndarray, conf_threshold: float) -> Tuple[Image.Image, List]:
        """Perform object detection on the image."""
        try:
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)
            
            # Run inference
            results = self.yolo_model.predict(source=pil_image)
            
            # Draw detections
            for result in results:
                for bbox in result.boxes:
                    confidence = bbox.conf[0].item()
                    if confidence >= conf_threshold:
                        self._draw_bbox(draw, bbox, confidence)
            
            return pil_image, results
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            raise

    def _draw_bbox(self, draw: ImageDraw, bbox, confidence: float):
        """Draw bounding box with confidence score."""
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Add confidence score with better visibility
        conf_text = f"Conf: {confidence:.2f}"
        draw.rectangle([x1, y1-20, x1+100, y1], fill="red")
        draw.text((x1+5, y1-15), conf_text, fill="white")

    def perform_ocr(self, image: np.ndarray, bbox: tuple) -> Tuple[Optional[str], Optional[float]]:
        """Perform OCR on a specific region of the image."""
        try:
            x1, y1, x2, y2 = bbox
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                logger.warning(f"Empty region detected: {bbox}")
                return None, None
                
            ocr_result = self.ocr_model.ocr(region, cls=True)
            
            if not ocr_result or not ocr_result[0]:
                return None, None
                
            texts = [line[1][0] for line in ocr_result[0]]
            confidences = [line[1][1] for line in ocr_result[0]]
            
            return " ".join(texts), float(np.mean(confidences))
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return None, None

class StreamlitUI:
    """Manages the Streamlit user interface."""
    
    def __init__(self):
        st.set_page_config(
            page_title="Advanced Object Detection",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Add main title with custom styling
        st.markdown("""
            <h1 style='text-align: center; color: #2e6c80; padding: 20px 0px; 
             border-radius: 10px; margin-bottom: 30px;'>
            License Plate & Container Number Detection 
            </h1>
            """, unsafe_allow_html=True)
        
        self.model_paths = {
            "License Plate": "yolov11_LP1.pt",
            "Container Detection": "yolov11n_Container_1Class.pt",
           #"License + Container": "Yolov11_LP+Container.pt"
        }

    def setup_sidebar(self) -> Tuple[str, float, Optional[BytesIO]]:
        """Configure and return sidebar elements."""
        st.sidebar.title("🛠️ Settings")
        
        # Model selection with custom styling
        #st.sidebar.markdown("### 📊 Model Selection")
        model_choice = st.sidebar.radio(
            "Select Detection Model",
            options=list(self.model_paths.keys()),
            help="Choose the type of detection to perform"
        )
        
        # Advanced settings
        #st.sidebar.markdown("### ⚙️ Advanced Settings")
        conf_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Adjust detection sensitivity"
        )
        
        # File upload with preview
        st.sidebar.markdown("### 📁 Image Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"],
            help="Support formats: JPG, JPEG, PNG"
        )
        
        return model_choice, conf_threshold, uploaded_file

    def display_results(
        self,
        original_image: Image.Image,
        result_image: Image.Image,
        bbox_coords: List,
        ocr_results: List,
        timing_info: Dict,
        uploaded_file_name: str  # Add this parameter
    ):
        """Display detection results and metrics."""
        # Display images in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📸 Original Image")
            st.image(original_image, use_container_width=True)
            
        with col2:
            st.markdown("### 🎯 Detection Results")
            st.image(result_image, use_container_width=True)
        
        # Performance metrics in single column
        st.markdown("### ⚡ Performance Metrics")
        metrics_cols = st.columns(3)
        with metrics_cols[0]:
            st.metric("Detection Time", f"{timing_info['detection']:.1f}ms")
        with metrics_cols[1]:
            st.metric("OCR Time", f"{timing_info['ocr']:.1f}ms")
        with metrics_cols[2]:
            st.metric("Total Time", f"{timing_info['total']:.1f}ms")
        
        # Display Bounding Box Coordinates
        st.markdown("### 📍 Bounding Box Coordinates")
        if bbox_coords:
            for idx, (x1, y1, x2, y2) in enumerate(bbox_coords, 1):
                st.text(f"Box {idx}: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
        else:
            st.warning("No bounding boxes detected")
        
        # Display OCR Results
        st.markdown("### 📝 OCR Results")
        if ocr_results:
            # Filter and display only rows with results
            df = pd.DataFrame(ocr_results)
            
            st.dataframe(
                df,
                use_container_width=True,
                height=min(len(df) * 60 + 40, 400)  # Dynamic height based on number of rows
            )
            
        else:
            st.warning("No text was detected in the image")

        # Export options
        st.markdown("### 💾 Export Options")
        col3, col4 = st.columns(2)
        with col3:
            base_name = os.path.splitext(uploaded_file_name)[0]
            detected_image_filename = f"{base_name}_detected.png"
        
            self._create_download_button(
                result_image,
                "Download Detected Image",
                detected_image_filename  # Use the custom filename
            )

        with col4:
            if ocr_results:
                df = pd.DataFrame(ocr_results)
                csv = df.to_csv(index=False).encode('utf-8')
                
                # Extract the base name of the uploaded file (without extension)
                base_name = os.path.splitext(uploaded_file_name)[0]
                csv_filename = f"{base_name}_ocr.csv"
                
                st.download_button(
                    "Download OCR Results",
                    csv,
                    csv_filename,
                    "text/csv"
                )

    @staticmethod
    def _create_download_button(image: Image.Image, label: str, filename: str):
        """Create a download button for images."""
        buf = BytesIO()
        image.save(buf, format="PNG")
        st.download_button(
            label=label,
            data=buf.getvalue(),
            file_name=filename,
            mime="image/png"
        )

def main():
    """Main application entry point."""
    try:
        # Initialize UI
        ui = StreamlitUI()
        model_choice, conf_threshold, uploaded_file = ui.setup_sidebar()
        
        # Initialize models
        if "model" not in st.session_state or st.session_state.get("model_choice") != model_choice:
            st.session_state.model = ModelManager.load_yolo_model(ui.model_paths[model_choice])
            st.session_state.model_choice = model_choice
        
        if "ocr_model" not in st.session_state:
            st.session_state.ocr_model = ModelManager.load_ocr_model()
        
        # Process image if uploaded
        if uploaded_file:
            processor = ImageProcessor(st.session_state.model, st.session_state.ocr_model)
            
            # Load and process image
            image_bytes = uploaded_file.read()
            image = np.array(Image.open(BytesIO(image_bytes)))
            
            # Perform detection and OCR
            start_time = time.time()
            result_image, results = processor.detect_objects(image, conf_threshold)
            detection_time = (time.time() - start_time) * 1000
            
            # Process OCR
            ocr_start = time.time()
            ocr_results = []
            bbox_coords = []
            
            for result in results:
                for bbox in result.boxes:
                    if bbox.conf[0].item() >= conf_threshold:
                        coords = tuple(map(int, bbox.xyxy[0]))
                        bbox_coords.append(coords)
                        text, conf = processor.perform_ocr(image, coords)
                        if text:
                            ocr_results.append({
                                "Detected Number": text,
                                "Confidence": f"{conf:.2f}",
                                "Location": f"({coords[0]}, {coords[1]})"
                            })
            
            ocr_time = (time.time() - ocr_start) * 1000
            total_time = detection_time + ocr_time
            
            # Display results
            ui.display_results(
                Image.open(BytesIO(image_bytes)),
                result_image,
                bbox_coords,
                ocr_results,
                {
                    "detection": detection_time,
                    "ocr": ocr_time,
                    "total": total_time
                },
                uploaded_file.name  # Pass the uploaded file name here
            )
        
        else:
            st.info("👆 Please upload an image to begin detection")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)

if __name__ == "__main__":
    main()

