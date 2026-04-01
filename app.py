import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from fruit_calories import fruit_data
import pandas as pd

# Page config
st.set_page_config(page_title="NutriScan", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #dfe9f3, #ffffff);
}

.title {
    text-align: center;
    font-size: 80px;
    font-weight: bold;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    text-align: center;
}

.metric {
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = YOLO("runs/detect/train3/weights/best.pt")

# Title
st.markdown('<p class="title">🌿 NutriScan</p>', unsafe_allow_html=True)
st.write("Upload food image to analyze nutrition ")

# Upload + Camera
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

with col2:
    camera_image = st.camera_input("Take Photo")

image = None
if uploaded_file:
    image = Image.open(uploaded_file)
elif camera_image:
    image = Image.open(camera_image)

if image:

    #  Resize image (max width 400px)
    image = image.resize((400, int(400 * image.height / image.width)))

    st.image(image, caption="Uploaded Image", width=400)

    img_array = np.array(image)
    results = model(img_array)

    fruit_counts = {}

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            fruit_name = model.names[cls]
            fruit_counts[fruit_name] = fruit_counts.get(fruit_name, 0) + 1

    if len(fruit_counts) == 0:
        st.warning("No fruits detected.")
    else:

        st.subheader(" Detected Fruits")

        total_calories = 0
        total_carbs = 0
        total_fat = 0

        for fruit, count in fruit_counts.items():
            if fruit in fruit_data:
                data = fruit_data[fruit]

                calories = data["calories"] * count
                total_calories += calories

                carbs = float(data["nutrients"].get("carbs", "0g").replace("g", "")) * count
                fat = float(data["nutrients"].get("fat", "0g").replace("g", "")) * count

                total_carbs += carbs
                total_fat += fat

        # Cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f'<div class="card"><p>Calories</p><p class="metric">{total_calories} kcal</p></div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="card"><p>Carbs</p><p class="metric">{total_carbs:.2f} g</p></div>', unsafe_allow_html=True)

        with col3:
            st.markdown(f'<div class="card"><p>Fat</p><p class="metric">{total_fat:.2f} g</p></div>', unsafe_allow_html=True)

        # Chart
        st.subheader(" Nutrition Breakdown")

        df = pd.DataFrame({
            "Nutrient": ["Calories", "Carbs", "Fat"],
            "Value": [total_calories, total_carbs, total_fat]
        })

        st.bar_chart(df.set_index("Nutrient"))

        # Details
        st.subheader(" Detailed Nutrition")

        for fruit, count in fruit_counts.items():
            if fruit in fruit_data:
                data = fruit_data[fruit]

                st.markdown(f"### {fruit.capitalize()}")
                st.write("Calories per fruit:", data["calories"])

                for nutrient, value in data["nutrients"].items():
                    st.write(f"- {nutrient}: {value}")