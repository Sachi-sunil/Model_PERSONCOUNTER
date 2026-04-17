from ultralytics import YOLO
import cv2
import os
import csv

# Load model
model = YOLO("yolov8n.pt")

# Paths
input_folder = "data/images"
output_folder = "outputs"

os.makedirs(output_folder, exist_ok=True)

MAX_CAPACITY = 5

results_summary = []

# Loop through images
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        results = model(image)

        person_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls == 0:  # person
                    person_count += 1

            annotated_frame = r.plot()

        # Overload check
        status = "OVERLOADED" if person_count > MAX_CAPACITY else "SAFE"

        print(f"{filename}: Count = {person_count}, Status = {status}")

        # Save output image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, annotated_frame)

        # Store results
        results_summary.append((filename, person_count, status))

print("\nProcessing Complete.")

with open("outputs/results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image Name", "Person Count", "Status"])
    writer.writerows(results_summary)

print("Results saved to outputs/results.csv")