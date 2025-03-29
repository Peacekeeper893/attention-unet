import streamlit as st
import pandas as pd
import numpy as np
import cv2

from evaluate import evaluate_single_image

from PIL import Image

st.html("""
    <style>
        .stMainBlockContainer {
            max-width:70rem;
        }
    </style>
    """
)

st.title('Retinal Vessel Segmentation DEMO')

st.text("Retinal vessel segmentation is a crucial step in analyzing retinal images for medical diagnosis. This project focuses on developing and evaluating segmentation techniques to accurately extract blood vessels from retinal fundus images. The primary objective is to study and implement modern deep learning-based approaches, including convolutional neural networks (CNNs) and U-Net architectures for Retinal Vessel Segmentation. ")


st.divider()

with st.container():
    st.subheader("Upload your retinal image data")
    st.write("Upload your retinal image and mask here.")

    with st.form(key='upload_form'):
        uploaded_image = st.file_uploader("Choose an image file (1)", type=["png", "jpg", "tif", "gif"])
        uploaded_mask = st.file_uploader("Choose a mask file (1)", type=["png", "jpg", "tif", "gif"])
        col1, col2 ,_,_ = st.columns(4)
        with col1:     
            submit_button = st.form_submit_button(label='Process Image', use_container_width=True, type='primary')
        with col2:
            example_button = st.form_submit_button(label='Use Example Image',type= 'secondary')

    if submit_button:
        if uploaded_image is None or uploaded_mask is None:
            st.error("Please upload both image and mask files.")
        else:
            # Read and decode the uploaded image using cv2
            image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)


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

    if example_button:
        # Load example image and mask
        example_image_path = (r'new_data\test\image\01_test_0.png')
        example_mask_path = r'new_data\test\mask\01_test_0.png'

        # Read and decode the example image using cv2
        example_image = cv2.imread(example_image_path, cv2.IMREAD_COLOR)

        # Read and decode the example mask
        example_mask = cv2.imread(example_mask_path, cv2.IMREAD_GRAYSCALE)

        if example_image is None or example_mask is None:
            st.error("Error loading example image or mask.")
            

        cat_images , image , mask = evaluate_single_image(example_image, example_mask)

        if cat_images is not None:
            st.divider()
            st.success("Example image processed successfully!")

            with st.container():
                st.subheader("Processed Example Image")
                st.image(cat_images, caption='Example Image , Example Mask and Predicted Results')
        else:
            st.error("Error processing the example image.")

st.divider()
st.subheader("About the Project")
st.write("This project is developed using PyTorch, a popular deep learning library. The segmentation model is based on the U-Net architecture and uses an attention mechanism to find useful features, which is widely used for image segmentation tasks. The project aims to provide an interactive platform for users to upload retinal images and obtain vessel segmentation results.")

st.write("More information and source code for the project can be found here: ")
st.link_button("GitHub" , "https://github.com/Peacekeeper893/attention-unet" ,icon="🔥" ,type='secondary')







       
        





