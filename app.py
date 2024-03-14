import streamlit as st
import torch
from PIL import Image
import cv2
import tempfile
import os
from fsrcnn_model import FSRCNNAdjusted  # Import your model definition here
from video_upscale import upscale_video  # Import your upscale function here if separate

# Initialize your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FSRCNNAdjusted(scale=2).to(device)  # Adjust parameters as necessary
model_path = 'F:\Tensorgo assignment\FSRCNN-x2.pt'  # Update this path
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

st.title("SD to HD Video Converter")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        # Write uploaded video to a temporary file
        tmpfile.write(uploaded_file.getvalue())
        input_video_path = tmpfile.name

    # Define output video path
    output_video_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")

    # Display the original video
    st.video(input_video_path)

    if st.button('Convert to HD'):
        with st.spinner('Processing...'):
            # Call your upscale function
            upscale_video(input_video_path, output_video_path, model, device, scale_factor=2)

        st.success('Conversion completed!')

        # Display and provide a download link for the converted video
        st.video(output_video_path)
        with open(output_video_path, 'rb') as file:
            st.download_button(label="Download HD Video", data=file, file_name="HD_video.mp4", mime='video/mp4')
