import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from fruit_calories import fruit_data
import pandas as pd
from nutrition_constants import daily_values

# Page config
st.set_page_config(page_title="Fruit Nutrition Analyzer", layout="wide")

# Load CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False

if "image_file" not in st.session_state:
    st.session_state.image_file = None

# Load model
model = YOLO("runs/detect/train3/weights/best.pt")

# Title Section
st.markdown('<div class="main-title"> Fruit Nutrition Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload or capture image to analyze nutrition</div>', unsafe_allow_html=True)

# Upload Section Logic
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

if not st.session_state.image_uploaded:

    # Centered container
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        subcol1, subcol2 = st.columns(2)

        with subcol1:
            uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        with subcol2:
            camera_image = st.camera_input(
                "📷 Camera",
                label_visibility="collapsed"
            )

        # Handle upload
        if uploaded_file:
            st.session_state.image_uploaded = True
            st.session_state.image_file = uploaded_file
            st.rerun()

        elif camera_image:
            st.session_state.image_uploaded = True
            st.session_state.image_file = camera_image
            st.rerun()

else:
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.markdown(
            f"""
            <div style="
                padding:12px;
                border-radius:10px;
                background:#f5f5f5;
                display:flex;
                justify-content:space-between;
                align-items:center;
            ">
                <span>📄 {st.session_state.image_file.name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("❌ Remove Image"):
            st.session_state.image_uploaded = False
            st.session_state.image_file = None
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Load image
image = None

if st.session_state.image_uploaded and st.session_state.image_file:
    image = Image.open(st.session_state.image_file)

# Process image
if image:

    image = image.resize((400, int(400 * image.height / image.width)))
    st.image(image, caption="Uploaded Image", width=300)

    results = model(np.array(image))

    fruit_counts = {}

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            fruit_name = model.names[cls]
            fruit_counts[fruit_name] = fruit_counts.get(fruit_name, 0) + 1

    if not fruit_counts:
        st.warning("No fruits detected.")
    else:

        total_calories = total_carbs = total_fat = total_fiber = 0
        total_vitamins = {"A": 0, "B": 0, "C": 0, "K": 0}
        total_minerals = {"potassium": 0}

        for fruit, count in fruit_counts.items():
            if fruit in fruit_data:
                data = fruit_data[fruit]
                nutrients = data["nutrients"]

                total_calories += data["calories"] * count
                total_carbs += nutrients.get("carbs", 0) * count
                total_fat += nutrients.get("fat", 0) * count
                total_fiber += nutrients.get("fiber", 0) * count

                for v, val in nutrients.get("vitamins", {}).items():
                    total_vitamins[v] += val * count

                for m, val in nutrients.get("minerals", {}).items():
                    total_minerals[m] += val * count

        #  Modern Cards
        st.markdown('<div class="section-title">Nutrition Overview</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        def card(title, value):
            return f"""
            <div class="card">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
            </div>
            """

        c1.markdown(card(" Calories", f"{total_calories} kcal"), unsafe_allow_html=True)
        c2.markdown(card(" Carbs", f"{total_carbs:.1f} g"), unsafe_allow_html=True)
        c3.markdown(card(" Fat", f"{total_fat:.1f} g"), unsafe_allow_html=True)
        c4.markdown(card(" Fiber", f"{total_fiber:.1f} g"), unsafe_allow_html=True)

        # Chart
        st.markdown('<div class="section-title">Nutrition Breakdown</div>', unsafe_allow_html=True)

        df = pd.DataFrame({
            "Nutrient": ["Calories", "Carbs", "Fat", "Fiber"],
            "Value": [total_calories, total_carbs, total_fat, total_fiber]
        })

        st.bar_chart(df.set_index("Nutrient"))

        # Daily %
        st.markdown('<div class="section-title">Daily Intake Coverage</div>', unsafe_allow_html=True)

        st.progress(min(int((total_calories / daily_values["calories"]) * 100), 100))
        st.write(f"Calories: {(total_calories / daily_values['calories']) * 100:.1f}%")

        # Fruits
        st.markdown('<div class="section-title">Detected Fruits</div>', unsafe_allow_html=True)

        for fruit, count in fruit_counts.items():
            st.markdown(f"""
            <div class="fruit-box">
                <b>{fruit.capitalize()}</b>
            </div>
            """, unsafe_allow_html=True)

        # Detailed Nutrition
        st.markdown('<div class="section-title">Detailed Nutrition</div>', unsafe_allow_html=True)

        for fruit, count in fruit_counts.items():
            if fruit in fruit_data:
                data = fruit_data[fruit]
                nutrients = data["nutrients"]

                st.markdown(f"###  {fruit.capitalize()}")

                st.write(f"Calories: {data['calories'] * count} kcal")
                st.write(f"Carbs: {nutrients.get('carbs', 0) * count} g")
                st.write(f"Fiber: {nutrients.get('fiber', 0) * count} g")
                st.write(f"Fat: {nutrients.get('fat', 0) * count} g")

                if "vitamins" in nutrients:
                    st.write("Vitamins:")
                    for v, val in nutrients["vitamins"].items():
                        st.write(f"- Vitamin {v}: {val * count}")

                if "minerals" in nutrients:
                    st.write("Minerals:")
                    for m, val in nutrients["minerals"].items():
                        st.write(f"- {m}: {val * count} mg")