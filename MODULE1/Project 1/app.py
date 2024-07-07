import streamlit as st
import tempfile
from ultralytics import YOLO
import pandas as pd
from io import BytesIO
import cv2
from PIL import Image


def predict(file):
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

    TRAINED_MODEL_PATH = './yolov10/runs/detect/train/weights/best.pt'
    model = YOLO(TRAINED_MODEL_PATH)
    results = model.predict(source=temp_file_path,
                            imgsz=640)
    annotated_img = results[0].plot()
    return annotated_img


st.title("Helmet Safety Detection")
st.header("Author: @tr.locne")
sample = st.button("Test")
st.divider()

col1, col2 = st.columns(2)
file = col1.file_uploader("Upload an image file for helmet detection")

check = col1.button("Check", type="primary")

if check and file is not None:
    pre = predict(file)
    pre1 = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
    col2.image(pre1, caption='Detected Image')
    st.balloons()
    st.success('This is a success message!', icon="✅")

elif check and file is None:
    st.warning("Please upload an image first.")

if sample:
    pt = cv2.imread(".\\yolov10\\Black-Workers-Need-a-Bill-of-Rights.jpeg")
    pre1 = cv2.cvtColor(pt, cv2.COLOR_BGR2RGB)
    TRAINED_MODEL_PATH = './yolov10/runs/detect/train/weights/best.pt'
    model = YOLO(TRAINED_MODEL_PATH)
    results = model.predict(source=pre1,
                            imgsz=640)
    pre1 = results[0].plot()
    col2.image(pre1, caption='Detected Image')
    st.balloons()
    st.success('This is a success message!', icon="✅")
