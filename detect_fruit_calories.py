from ultralytics import YOLO
from fruit_calories import fruit_data

# Load model
model = YOLO("runs/detect/train3/weights/best.pt")

# Test image
image_path = "test.jpg"

# Run detection
results = model(image_path, conf=0.25, save=True)

fruit_counts = {}

print("\nDetected Fruits:\n")

for r in results:
    if r.boxes is None:
        continue

    for box in r.boxes:
        cls = int(box.cls[0])
        fruit_name = model.names[cls]

        fruit_counts[fruit_name] = fruit_counts.get(fruit_name, 0) + 1

total_calories = 0

for fruit, count in fruit_counts.items():

    if fruit in fruit_data:
        data = fruit_data[fruit]
        calories = data["calories"] * count

        print(f"Fruit: {fruit}")
        print(f"Count: {count}")
        print(f"Calories per fruit: {data['calories']}")
        print(f"Total Calories: {calories}")

        print("Nutrients:")
        for nutrient, value in data["nutrients"].items():
            print(f"- {nutrient}: {value}")

        print()

        total_calories += calories

print("TOTAL CALORIES:", total_calories)

print("\nAnnotated image saved in:")
print("runs/detect/predict/")