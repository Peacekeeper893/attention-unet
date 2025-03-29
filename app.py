import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import cv2

from evaluate import evaluate_single_image

from PIL import Image
st.title('Retinal Vessel Segmentation DEMO')

st.text("Retinal vessel segmentation is a crucial step in analyzing retinal images for medical diagnosis. This project focuses on developing and evaluating segmentation techniques to accurately extract blood vessels from retinal fundus images. The primary objective is to study and implement modern deep learning-based approaches, including convolutional neural networks (CNNs) and U-Net architectures for Retinal Vessel Segmentation. ")


st.divider()

with st.container():
    st.subheader("Upload your retinal image data")
    st.write("Upload your retinal image and mask here.")

    with st.form(key='upload_form'):
        uploaded_image = st.file_uploader("Choose an image file (1)", type=["png", "jpg", "tif", "gif"])
        uploaded_mask = st.file_uploader("Choose a mask file (1)", type=["png", "jpg", "tif", "gif"])
        submit_button = st.form_submit_button(label='Process Image')

    if submit_button:
        if uploaded_image is None or uploaded_mask is None:
            st.error("Please upload both image and mask files.")
        else:
            # Read and decode the uploaded image using cv2
            image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
            # Convert BGR to RGB
  

            image_real = cv2.imread('new_data\\test\image\\01_test_0.png', cv2.IMREAD_COLOR)


            

            # Read and decode the uploaded mask
            mask_bytes = uploaded_mask.read()
            mask = Image.open(uploaded_mask)
            mask = np.array(mask)

            cat_images , image , mask = evaluate_single_image(image, mask)

            if cat_images is not None:
                st.divider()
                st.success("Image processed successfully!")

                with st.container():
                    st.subheader("Processed Image")
                    st.image(cat_images, caption='Uploaded Image , Uploaded Mask and Predicted Results')
            else:
                st.error("Error processing the image. Please check the uploaded files.")






       
        





