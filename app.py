import streamlit as st
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_bytes
import pandas as pd
import tempfile
import os

def extract_red_text(image):
    """ Extracts red-colored text from an image using color filtering and OCR. """
    try:
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 70, 50])   # Adjusted to include darker reds
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 70, 50])  # Adjusted to capture more variations
        upper_red2 = np.array([180, 255, 255])

        # Optional: Add a middle red range if needed
        lower_red3 = np.array([10, 70, 50])  
        upper_red3 = np.array([20, 255, 255])  

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask3 = cv2.inRange(hsv, lower_red3, upper_red3)

        red_mask = mask1 + mask2 + mask3  # Combine all masks

        
        red_text_only = cv2.bitwise_and(open_cv_image, open_cv_image, mask=red_mask)
        gray = cv2.cvtColor(red_text_only, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        custom_config = "--psm 6"
        extracted_text = pytesseract.image_to_string(thresh, config=custom_config)
        
        return extracted_text.strip()
    except Exception as e:
        return f"Error extracting red text: {e}"

def process_pdf(pdf_bytes):
    """ Converts PDF to images, extracts red text, and returns results as a DataFrame. """
    images = convert_from_bytes(pdf_bytes)
    data = []
    
    for i, img in enumerate(images):
        red_text = extract_red_text(img)
        data.append([f"Page {i+1}", red_text])
    
    df = pd.DataFrame(data, columns=["Page", "Red Text"])
    return df

st.title("Red Text Extractor from PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    with st.spinner("Processing PDF..."):
        pdf_bytes = uploaded_file.read()
        df = process_pdf(pdf_bytes)
        st.success("Extraction complete!")
        st.dataframe(df)
        
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        df.to_excel(output_file.name, index=False, engine='openpyxl')
        
        with open(output_file.name, "rb") as f:
            st.download_button("Download Extracted Text", f, "extracted_text.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        os.unlink(output_file.name)
