import torch
import cv2
from torchvision import models, transforms

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

image_path = "test.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

input_tensor = transform(image_rgb)

input_tensor = input_tensor.unsqueeze(0)

with torch.no_grad():
    outputs = model(input_tensor)

# Count persons
person_count = 0

for i in range(len(outputs[0]['labels'])):
    label = outputs[0]['labels'][i].item()
    score = outputs[0]['scores'][i].item()

    # COCO class 1 = person
    if label == 1 and score > 0.5:
        person_count += 1

print("People Count:", person_count)

for i in range(len(outputs[0]['boxes'])):
    label = outputs[0]['labels'][i].item()
    score = outputs[0]['scores'][i].item()

    if label == 1 and score > 0.5:
        box = outputs[0]['boxes'][i].numpy().astype(int)

        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Faster R-CNN Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

MAX_CAPACITY = 5

if person_count > MAX_CAPACITY:
    print("OVERLOADED")
else:
    print("SAFE")