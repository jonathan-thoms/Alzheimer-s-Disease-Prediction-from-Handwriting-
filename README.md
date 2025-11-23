# 🧠 Alzheimer’s Disease Prediction via Handwriting Analysis

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-green)

A Deep Learning classification system designed to detect early signs of Alzheimer's disease by analyzing handwriting kinematic features. This project utilizes the **DARWIN dataset** and a custom-tuned Neural Network to classify subjects as "Healthy" or "Alzheimer's Patient" based on numeric feature extraction.

## 🔬 Project Overview

Early detection of Alzheimer's is critical for patient care. This project automates the analysis of handwriting tasks (drawing, writing) by processing kinematic data (velocity, pressure, jerk).



### Key Features
-   **Deep Learning Classifier:** A 3-layer Feed-Forward Neural Network implemented in TensorFlow/Keras.
-   **Robust Preprocessing:** Automated scaling (MinMax) and label encoding pipeline using `joblib` for reproducibility.
-   **Regularization:** Implements `Dropout` layers to prevent overfitting on the tabular dataset.
-   **Inference Engine:** Includes a standalone prediction script to process new patient data.

## 📂 Dataset

The model is trained on the **DARWIN (Diagnosis Alzheimer’s Disease for Researchers and Doctors)** dataset.
* **Input:** Numeric features extracted from handwriting tasks.
* **Target:** Binary classification (Healthy vs. Patient).

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/alzheimers-handwriting-ai.git](https://github.com/yourusername/alzheimers-handwriting-ai.git)
    cd alzheimers-handwriting-ai
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Data:**
    Place your `data.csv` file inside the `data/` directory.

## 🚀 Usage

### 1. Training the Model
Run the training pipeline. This will preprocess data, train the network, and save the model artifacts (`.keras`, `scaler.pkl`, `encoder.pkl`).

```bash
python train.py
