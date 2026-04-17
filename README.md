# Lift Overload Detection using Image and Video Analytics

## 📌 Project Overview

This project focuses on detecting and counting the number of people in a lift using computer vision techniques. The system identifies whether a lift is overloaded by comparing the detected number of people with a predefined capacity threshold.

The project explores multiple deep learning architectures to analyze their performance in terms of accuracy, speed, and efficiency.

---

## 🎯 Objective

* Detect people in images/videos of a lift
* Count the number of individuals present
* Determine overload condition based on threshold
* Compare different deep learning architectures

---

## 🧠 Architectures Used

### 1. YOLO (You Only Look Once)

* Type: Single-stage object detector
* Advantage: Fast and efficient
* Use: Real-time people detection and counting

### 2. Faster R-CNN

* Type: Two-stage object detector
* Advantage: High accuracy
* Use: Precise detection and comparison with YOLO

### 3. MobileNet SSD

* Type: Lightweight detector
* Advantage: Low computational cost
* Use: Efficient detection on low-resource systems

### 4. CNN (Baseline Model)

* Type: Image classification
* Use: Classifies images into:

  * Low Crowd
  * High Crowd
* Limitation: Cannot count individual people

---

## 🧩 System Pipeline

Input Image/Video
↓
Object Detection Model (YOLO / Faster R-CNN / SSD)
↓
Person Detection
↓
Counting People
↓
Compare with Threshold
↓
Output: SAFE / OVERLOADED

---

## ⚙️ Technologies Used

* Python
* OpenCV
* PyTorch
* TensorFlow/Keras
* Ultralytics YOLO

---

## 📂 Project Structure

```
people_counting_project/
│
├── data/
│   ├── images/
│   └── cnn/
│
├── models/
│
├── outputs/
│   ├── images/
│   └── results.csv
│
├── batch_process.py
├── faster_rcnn.py
├── mobilenet_ssd.py
├── cnn_model.py
├── cnn_predict.py
└── README.md
```

---

## 🚀 How to Run

### 1. Install Dependencies

```
pip install ultralytics opencv-python torch torchvision tensorflow numpy
```

### 2. Run YOLO Detection

```
python batch_process.py
```

### 3. Run Faster R-CNN

```
python faster_rcnn.py
```

### 4. Run MobileNet SSD

```
python mobilenet_ssd.py
```

### 5. Train CNN Model

```
python cnn_model.py
```

### 6. Test CNN Model

```
python cnn_predict.py
```

---

## 📊 Evaluation Metrics

* Person Count Accuracy
* Mean Absolute Error (MAE) *(if ground truth available)*
* Detection Speed (Inference Time)

---

## 📈 Observations

| Model         | Speed     | Accuracy  | Remarks             |
| ------------- | --------- | --------- | ------------------- |
| YOLO          | High      | Good      | Best balance        |
| Faster R-CNN  | Low       | Very High | Most accurate       |
| MobileNet SSD | High      | Moderate  | Lightweight         |
| CNN           | Very High | Low       | Only classification |

---

## ⚠️ Limitations

* CNN cannot perform object detection
* Performance may drop in crowded or occluded scenes
* Pretrained models may misclassify non-human objects

---

## 🔮 Future Scope

* Integrate weight estimation for accurate overload detection
* Deploy in real-time CCTV systems
* Use depth sensors for better accuracy
* Optimize models for edge devices

---

## ✅ Conclusion

The project successfully demonstrates people detection and overload identification using multiple deep learning architectures. YOLO provides the best trade-off between speed and accuracy, while Faster R-CNN achieves higher precision. MobileNet SSD offers an efficient lightweight alternative.

---

## 👤 Author

Sachin Sunil
