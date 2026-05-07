# Fruit Nutrition Analyzer with AI Recommendations

A smart web application that detects fruits from images and provides detailed nutritional insights along with intelligent dietary recommendations.

---

## Features

- Fruit detection using YOLO (image upload + camera)
- Nutritional analysis (calories, carbs, fiber, fat)
- AI-based recommendations with fallback logic
- BMI calculation with health category
- Clean and interactive Streamlit UI

---

## Technologies Used

- Python  
- Streamlit  
- YOLO (Ultralytics)  
- NumPy  
- Pandas  
- Google Generative AI (Gemini API)

---

## Project Structure

fruit-nutrition-analyzer/
│
├── app.py                  # Main Streamlit app
├── utils.py                # AI recommendation & helper functions
├── model/                  # Trained YOLO model (best.pt)
├── dataset/                # Training dataset (images + labels)
├── requirements.txt        # Dependencies
└── README.md               # Project documentation


---

## How It Works

1. Upload image or capture from camera  
2. Model detects fruits  
3. Nutritional values are calculated  
4. BMI is computed (if user inputs data)  
5. AI generates recommendations (or fallback logic runs)

---

## Model Training

- Dataset: Kaggle  
- Model: YOLOv8  
- Epochs: 80  
- Output: best.pt (trained weights)

---

## Evaluation

- Metric: mAP (Mean Average Precision)  
- Accuracy based on detection performance  

---

## Future Scope

- Add more food categories  
- Real-time video detection  
- Personalized diet planning  
- Cloud deployment  

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fruit-nutrition-analyzer.git
cd fruit-nutrition-analyzer

pip install -r requirements.txt

streamlit run app.py