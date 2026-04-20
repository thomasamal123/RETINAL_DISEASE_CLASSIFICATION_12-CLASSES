# Eye Disease Classification using Deep Learning and Machine Learning

## Project Overview

This project focuses on automated classification of retinal fundus images into multiple eye disease categories using deep learning models, along with a comparison against traditional machine learning approaches. The objective is to assist ophthalmologists in early diagnosis, reduce manual screening time, and provide an accessible AI-based diagnostic support system through a web application.

---

## Objectives

* Classify retinal fundus images into 12 eye disease categories
* Compare deep learning models with traditional machine learning algorithms
* Improve interpretability using Grad-CAM visualizations
* Deploy a web-based application for real-time predictions
* Generate downloadable PDF reports

---

## Methodology

### Data Collection and Preprocessing

* Collected a labeled dataset of retinal fundus images
* Performed preprocessing:

  * Image resizing and normalization
  * Noise reduction
* Applied data augmentation:

  * Rotation
  * Flipping
  * Brightness adjustment
* Addressed class imbalance and improved model generalization

---

### Deep Learning Models

The following architectures were implemented and evaluated:

* EfficientNet-B0
* EfficientNet-B3
* EfficientNet-B4
* ResNet50
* DenseNet121

---

### Ensemble Learning

* Combined top-performing models (EfficientNet-B4 and DenseNet121)
* Used probability averaging for predictions
* Improved overall classification performance and robustness

---

### Traditional Machine Learning Comparison

* Extracted deep features from trained CNN models
* Trained classical machine learning models:

  * Support Vector Machine (SVM)
  * Random Forest (RF)
  * K-Nearest Neighbors (KNN)
  * Logistic Regression
* Compared results with deep learning models

---

### Model Evaluation

Performance was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

### Explainability

* Implemented Grad-CAM for visual explanations
* Highlighted important regions in retinal images
* Improved transparency of model predictions

---

### Web Application

* Developed using Streamlit
* Features:

  * Image upload
  * Real-time prediction
  * Grad-CAM visualization
  * PDF report generation

---

## Results Summary

* Deep learning models achieved the highest performance
* EfficientNet-B4 and DenseNet121 performed best among individual models
* Traditional machine learning models performed moderately
* Ensemble model provided the most reliable predictions

---

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy, Pandas
* Matplotlib, Seaborn
* Streamlit
* Grad-CAM

---

## How to Run

### Clone the Repository

```bash
git clone https://github.com/your-username/eye-disease-classification.git
cd eye-disease-classification
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## Project Structure

```
├── data/                 
├── models/               
├── notebooks/            
├── app.py                
├── utils/                
├── reports/              
├── requirements.txt      
└── README.md             
```

---

## Applications

* Early detection of eye diseases
* Clinical decision support systems
* Remote healthcare accessibility
* Medical AI research

---

## Disclaimer

This project is intended for educational and research purposes only and should not be used for medical diagnosis.

---

## Author

Amal Thomas
