import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from sklearn.cluster import KMeans
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import zipfile

# Enable native HEIC/HEIF photo support to prevent upload crashes
register_heif_opener()

# Page configuration setup
st.set_page_config(page_title="Paint-by-Numbers Studio", page_icon="🎨", layout="centered")

# --- SAFE IMAGE CONVERTER & PROCESSOR ---
def process_image_to_data(image_bytes, num_colors=24):
    try:
        # Load image via PIL to safely support JPG, PNG, and HEIC formats
        pil_img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB color mode if image is transparent PNG (RGBA) or grayscale
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
            
        img = np.array(pil_img)
        h, w, _ = img.shape
        
        # Scale image safely to avoid memory/timeout issues
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w, _ = img.shape

        # Color Quantization using K-Means Clustering
        pixels = img.reshape((-1, 3))
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=3)
        labels = kmeans.fit_predict(pixels)
        colors = kmeans.cluster_centers_.astype(np.uint8)
        
        label_matrix = labels.reshape((h, w))
        quantized_img = colors[label_matrix]
        
        # Trace contours and build labeled points map
        contour_data = []
        for color_idx in range(num_colors):
            color_mask = np.zeros((h, w), dtype=np.uint8)
            color_mask[label_matrix == color_idx] = 255
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
            
            contours, _ = cv2.findContours(color_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 60: 
                    continue
                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    contour_data.append({
                        "contour": contour,
                        "label": str(color_idx + 1),
                        "center": (cX, cY)
                    })
                    
        return quantized_img, contour_data, colors, w, h
    except Exception as e:
        raise RuntimeError(f"Error handling image processing: {str(e)}")

# --- PRINTABLE VECTOR PDF BUILDER ---
def generate_pdf_blueprint(w, h, contour_data, colors):
    pdf_buffer = io.BytesIO()
    page_w, page_h = letter
    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
    
    # Calculate perfect scaling margins
    scale_w = (page_w - 60) / w
    scale_h = (page_h - 120) / h
    scale = min(scale_w, scale_h)
    
    offset_x = 30
    offset_y = 100
    
    # Page 1: Line Art Blueprint Map
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(30, page_h - 40, "Your Paint-by-Numbers Canvas Outline")
    pdf.setStrokeColorRGB(0.6, 0.6, 0.6)
    pdf.setLineWidth(0.5)
    
    for item in contour_data:
        points = item["contour"].reshape(-1, 2)
        if len(points) < 2: 
            continue
        
        path = pdf.beginPath()
        path.moveTo(offset_x + points[0][0] * scale, offset_y + (h - points[0][1]) * scale)
        for pt in points[1:]:
            path.lineTo(offset_x + pt[0] * scale, offset_y + (h - pt[1]) * scale)
        path.close()
        pdf.drawPath(path, stroke=1, fill=0)
        
        # Draw centering digits safely
        cx, cy = item["center"]
        pdf.setFont("Helvetica", 6)
        pdf.setFillColorRGB(0.4, 0.4, 0.4)
        pdf.drawCentredString(offset_x + cx * scale, offset_y + (h - cy) * scale, item["label"])
        
    # Page 2: Acrylic Color Swatch Index Key Map
    pdf.showPage()
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(30, page_h - 50, "Color Palette Swatch Key Guide")
    
    box_y = page_h - 100
    for idx, color in enumerate(colors):
        pdf.setFillColorRGB(color[0]/255.0, color[1]/255.0, color[2]/255.0)
        pdf.rect(30, box_y, 40, 20, fill=1, stroke=1)
        
        pdf.setFillColorRGB(0, 0, 0)
        pdf.setFont("Helvetica", 11)
        pdf.drawString(85, box_y + 5, f"Jar #{idx + 1:02d} — Hex: #{color[0]:02x}{color[1]:02x}{color[2]:02x} — RGB: {list(color)}")
        box_y -= 30
        
    pdf.save()
    pdf_buffer.seek(0)
    return pdf_buffer

# --- STREAMLIT UI VIEW APPLICATION LAYER ---
st.title("🎨 Custom Paint by Number Generator")
st.write("Convert any photo directly into a scalable paint-by-numbers outline layout template.")

# Configuration Inputs Drawer panel
st.sidebar.header("🔧 Options Panel")
num_colors = st.sidebar.slider("Number of Paint Colors", min_value=8, max_value=48, value=24, step=2)

# File Drag & Drop frame handler
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "heic"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image_bytes, use_container_width=True)
        
    if st.button("Generate Template Design Pack", type="primary", use_container_width=True):
        with st.spinner("Analyzing image contours and building palette colors..."):
            try:
                # Trigger the error-insulated engine
                quantized_img, contour_data, colors, w, h = process_image_to_data(image_bytes, num_colors)
                
                with col2:
                    st.subheader("Target Preview Map")
                    st.image(quantized_img, use_container_width=True)
                
                # Create vector PDF file streams
                pdf_file = generate_pdf_blueprint(w, h, contour_data, colors)
                
                # Create reference preview JPEG image buffers 
                preview_io = io.BytesIO()
                Image.fromarray(quantized_img).save(preview_io, format="JPEG")
                
                # Bundle assets securely into an in-memory ZIP package download
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    zip_file.writestr("colored_target_preview.jpg", preview_io.getvalue())
                    zip_file.writestr("printable_canvas_outline.pdf", pdf_file.getvalue())
                zip_buffer.seek(0)
                
                st.success("🎉 Blueprint Kit compiled cleanly without error closures!")
                
                st.download_button(
                    label="📥 Download Kit Archive (.ZIP)",
                    data=zip_buffer,
                    file_name="my_custom_paint_kit.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            except Exception as processing_error:
                st.error(f"Execution Error: {str(processing_error)}")
