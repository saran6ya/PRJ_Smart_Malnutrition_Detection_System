
from ultralytics import YOLO

# Load trained model
model = YOLO("models/best_malnutrition_yolov8n.pt")

# Display model information
print("Model loaded successfully!")

print("\nAvailable Classes:")

for class_id, class_name in model.names.items():
    print(class_id, ":", class_name)
