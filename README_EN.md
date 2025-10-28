# 🥬 Lettuce Health Classification (InceptionV3)

## 📘 Introduction  
This project focuses on **classifying the health status of lettuce plants (Healthy / Unhealthy)** using a pretrained **InceptionV3** model.  
The goal is to **automatically detect healthy vs. diseased plants** to support smart agriculture.

---

## 📂 Project Structure  

```
lettuce_classification/
├── data/                    # Dataset directory (not included)
├── notebooks/               # Training & evaluation notebooks
├── outputs/                 # Saved models and results
├── src/                     # Main source code
├── requirements.txt         # Required dependencies
└── README.md
```

---

## 📦 Environment Setup

Requirement: **Python ≥ 3.10**

```bash
git clone https://github.com/HuySang-04/lettuce_classification.git
cd lettuce_classification
pip install -r requirements.txt
```

---

## 🧠 Dataset & Pretrained Model  

- **Dataset:** Google Drive link (not included)
- **Pretrained Model:** Google Drive link (not included)

> After downloading, place the dataset into the `data/` folder and the model into the `outputs/` folder.

---

## 🚀 Train the Model  

```bash
python src/lettuce_health/lettuce_health_train.py
```

---

## 🔍 Test the Model  

```bash
python src/lettuce_health/lettuce_health_test.py
```

---

## 📊 Training Results  

| Metric | Value |
|:-------|:------|
| Train Accuracy | ≈ 99% |
| Validation Accuracy | ≈ 99.5% |
| Loss decreases stably | ✅ |

<div align="center">

### **Accuracy & Loss Curves**  
<img src="outputs/figures/accuracy_loss.png" width="700">
</div>

---

### 📊 Test Results

<div align="center">

### **Accuracy & Classification Report**  
<img src="outputs/figures/test_accuracy.png" width="400">

### **Confusion Matrix**  
<img src="outputs/figures/confusion_matrix.png" width="400">
</div>

---

## 🖼️ Demo Screenshots

![healthhy](./outputs/demo_web/demo1.png)

![unhealthhy](./outputs/demo_web/demo2.png)

---

## ⚙️ Recommended Environment  

- Python 3.10+
- TensorFlow / Keras
- NumPy  
- Matplotlib  
- scikit-learn  
- GPU support (optional but recommended)

---

## 📄 Notes  

- The **InceptionV3** model is fine-tuned on real lettuce images.
- Can be extended to other crops or multiple plant diseases.
- All required libraries are listed in `requirements.txt`.

---

⭐ If you find this project helpful, please consider giving it a **star 🌟** on GitHub!
