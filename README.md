Project Title: Eye Disease Classification using Deep Learning and Traditional Machine Learning

Author: AMAL THOMAS

Project Overview: This project focuses on automated detection of multiple eye diseases from retinal fundus images using state-of-the-art deep learning models and a comparison with traditional machine learning approaches. The aim is to assist ophthalmologists in early diagnosis, reduce manual screening time, and make disease detection more accessible through a web-based application.

The system classifies fundus images into multiple disease categories, provides heatmap visualizations (Grad-CAM) to highlight regions of interest, and generates downloadable PDF reports for users.

What I Did in This Project:

Collected and preprocessed a dataset of labeled fundus images for 12 eye disease categories.
Applied data augmentation (rotation, flipping, brightness changes) to address class imbalance.
Trained and evaluated multiple deep learning architectures.
Implemented Grad-CAM visualizations for model interpretability.
Built a web-based application for user-friendly predictions and reports.
Compared deep learning performance with traditional machine learning classifiers using CNN-based feature extraction.
Deep Learning Models Used:

EfficientNet-B0
EfficientNet-B3
EfficientNet-B4
ResNet50
DenseNet121
Ensemble Model:

Combined the top two performing models into an ensemble by averaging their prediction probabilities, improving overall classification accuracy.
Traditional Machine Learning Comparison:

Extracted deep features from two of the CNN models.
Used these features to train and evaluate traditional ML classifiers such as Support Vector Machine (SVM), Random Forest (RF), K-Nearest Neighbors (KNN), and Logistic Regression.
Compared performance metrics (accuracy, precision, recall, F1-score) between deep learning models and ML classifiers.
Comparison Summary:

Deep learning models, especially EfficientNet-B4 and DenseNet121, achieved the highest accuracy due to their ability to learn complex features from raw images.
Traditional ML models performed well on extracted features but were limited by the quality of feature extraction and did not match top CNN performance.
Ensemble learning further improved prediction reliability.
How This Project is Useful:

Early Detection: Identifies eye diseases before progression, potentially preventing vision loss.
Healthcare Support: Provides a decision-support tool for ophthalmologists.
Transparency: Grad-CAM visualizations improve trust by showing the decision-making process.
Scalability: Web deployment allows use in remote or underserved areas.
Educational Value: Demonstrates practical comparison between cutting-edge deep learning and traditional ML techniques in medical imaging.
Technologies Used:

Python, TensorFlow/Keras
OpenCV, NumPy, Pandas
Matplotlib, Seaborn
Streamlit (for web app)
Grad-CAM for explainability
