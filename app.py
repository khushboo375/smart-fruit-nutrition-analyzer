import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from fruit_calories import fruit_data
import pandas as pd
from nutrition_constants import daily_values

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
        total_fiber = 0
        total_vitamins = {"A": 0, "B": 0, "C": 0, "K": 0}
        total_minerals = {"potassium": 0}

        for fruit, count in fruit_counts.items():
            if fruit in fruit_data:
                data = fruit_data[fruit]

                total_calories += data["calories"] * count

                nutrients = data["nutrients"]

                total_carbs += nutrients.get("carbs", 0) * count
                total_fat += nutrients.get("fat", 0) * count
                total_fiber += nutrients.get("fiber", 0) * count

                for v, val in nutrients.get("vitamins", {}).items():
                    total_vitamins[v] += val * count

                for m, val in nutrients.get("minerals", {}).items():
                    total_minerals[m] += val * count

        # Cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f'<div class="card"><p>Calories</p><p class="metric">{total_calories} kcal</p></div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="card"><p>Carbs</p><p class="metric">{total_carbs:.2f} g</p></div>', unsafe_allow_html=True)

        with col3:
            st.markdown(f'<div class="card"><p>Fat</p><p class="metric">{total_fat:.2f} g</p></div>', unsafe_allow_html=True)

        with col4:
            st.markdown(f'<div class="card"><p>Fiber</p><p class="metric">{total_fiber:.2f} g</p></div>', unsafe_allow_html=True)

        # Chart
        st.subheader(" Nutrition Breakdown")

        df = pd.DataFrame({
            "Nutrient": ["Calories", "Carbs", "Fat"],
            "Value": [total_calories, total_carbs, total_fat]
        })

        st.bar_chart(df.set_index("Nutrient"))

        #Daily % Section
        st.subheader(" Daily Nutrition Coverage (%)")

        calorie_pct = (total_calories / daily_values["calories"]) * 100
        carb_pct = (total_carbs / daily_values["carbs"]) * 100
        fat_pct = (total_fat / daily_values["fat"]) * 100
        fiber_pct = (total_fiber / daily_values["fiber"]) * 100

        vitamin_c_pct = (total_vitamins["C"] / daily_values["vitamin_c"]) * 100 if total_vitamins["C"] else 0
        vitamin_a_pct = (total_vitamins["A"] / daily_values["vitamin_a"]) * 100 if total_vitamins["A"] else 0
        potassium_pct = (total_minerals["potassium"] / daily_values["potassium"]) * 100 if total_minerals["potassium"] else 0

        st.write(f"Calories: {calorie_pct:.1f}%")
        st.write(f"Carbs: {carb_pct:.1f}%")
        st.write(f"Fat: {fat_pct:.1f}%")
        st.write(f"Fiber: {fiber_pct:.1f}%")
        st.write(f"Vitamin C: {vitamin_c_pct:.1f}%")
        st.write(f"Vitamin A: {vitamin_a_pct:.1f}%")
        st.write(f"Potassium: {potassium_pct:.1f}%")

        # Details
        st.subheader(" Detailed Nutrition")

        for fruit, count in fruit_counts.items():
            if fruit in fruit_data:
                data = fruit_data[fruit]
                nutrients = data["nutrients"]

                st.markdown(f"### {fruit.capitalize()}")

                st.write(f"Calories: {data['calories'] * count} kcal")
                st.write(f"Carbs: {nutrients.get('carbs', 0) * count} g")
                st.write(f"Fiber: {nutrients.get('fiber', 0) * count} g")
                st.write(f"Fat: {nutrients.get('fat', 0) * count} g")

                # Vitamins
                st.write("Vitamins:")
                for v, val in nutrients.get("vitamins", {}).items():
                    st.write(f"- Vitamin {v}: {val * count}")

                # Minerals
                st.write("Minerals:")
                for m, val in nutrients.get("minerals", {}).items():
                    st.write(f"- {m}: {val * count} mg")