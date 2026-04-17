import cv2
import numpy as np

# Load model
net = cv2.dnn.readNetFromCaffe(
    "models/MobileNetSSD_deploy.prototxt",
    "models/MobileNetSSD_deploy.caffemodel"
)

# Class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Load image
image = cv2.imread("test.jpg")
(h, w) = image.shape[:2]

# Preprocess
blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)

net.setInput(blob)
detections = net.forward()

person_count = 0

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        idx = int(detections[0, 0, i, 1])

        if CLASSES[idx] == "person":
            person_count += 1

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

print("People Count:", person_count)

cv2.imshow("MobileNet SSD", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

MAX_CAPACITY = 5

if person_count > MAX_CAPACITY:
    print("OVERLOADED")
else:
    print("SAFE")