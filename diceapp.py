import streamlit as st
import cv2
import os
import glob
from PIL import Image
import numpy as np
from ultralytics import YOLO

# App title
st.set_page_config(page_title="🎲 Two Dice Game", layout="centered")
st.title("🎲 Two Dice Game using YOLOv8 Object Detection")

# Load YOLOv8 model
model_path = 'D:/DL_DEMO/Assignment3/DL_3/yolov8n.pt'
model = YOLO(model_path)

# Load test images
image_folder = 'D:/DL_DEMO/Assignment3/DL_3/dice_dataset/test/images'
image_paths = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))

# Game state using session state
if 'index' not in st.session_state:
    st.session_state.index = 0
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'game_over' not in st.session_state:
    st.session_state.game_over = False

# Score button handler
def handle_score():
    if st.session_state.index < len(image_paths):
        img_path = image_paths[st.session_state.index]
        img = cv2.imread(img_path)
        results = model(img)[0]

        dice_values = []
        for box in results.boxes:
            cls = int(box.cls[0]) + 1  # class index + 1
            dice_values.append(cls)

        # Only top 2 dice values
        dice_values = dice_values[:2]

        annotated_img = results.plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        if len(dice_values) == 2:
            if dice_values[0] == dice_values[1]:
                st.session_state.game_over = True
                st.session_state.message = f"🎯 Game Over! Final Score: {st.session_state.score}"
            else:
                round_score = sum(dice_values)
                st.session_state.score += round_score
                st.session_state.message = f"✅ Score this round: {round_score} | Total Score: {st.session_state.score}"
        else:
            st.session_state.message = "⚠️ Could not detect 2 dice."

        st.session_state.annotated_image = annotated_img
        st.session_state.index += 1

# Show image and buttons
if st.session_state.index < len(image_paths) and not st.session_state.game_over:
    st.image(image_paths[st.session_state.index], caption=f"Image {st.session_state.index + 1}", use_column_width=True)
    st.button("🎯 Score This Round", on_click=handle_score)
else:
    st.write("✅ All images processed or game ended.")

# Display result
if 'annotated_image' in st.session_state:
    st.image(st.session_state.annotated_image, caption="Detected Dice", use_column_width=True)
if 'message' in st.session_state:
    st.success(st.session_state.message)

# Reset button
if st.button("🔄 Reset Game"):
    st.session_state.index += 1
    st.session_state.score = 0
    st.session_state.game_over = False
    st.session_state.message = ""
    if 'annotated_image' in st.session_state:
        del st.session_state.annotated_image
