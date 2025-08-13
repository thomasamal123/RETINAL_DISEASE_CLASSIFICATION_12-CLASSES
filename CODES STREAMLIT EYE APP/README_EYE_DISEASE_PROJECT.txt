
README
Eye Disease Detection using CNN Models and Streamlit App
Name: Amal Thomas


---

Project Overview
This project is about detecting eye diseases from fundus images using deep learning models.
I used two models, EfficientNet-B3 and EfficientNet-B4, trained separately and then combined as an ensemble for better accuracy.
A Streamlit web app was developed to make predictions, show Grad-CAM heatmaps, and generate a PDF medical report.

---

Folder Structure

- CODES EYE DISEASE CLASSIFICATION AND STREAMLIT
  - EYE DISEASE CLASSIFICATION.ipynb // Training CNN models and preparing the dataset in notebook.
  - EYE DISEASE CLASSIFICATION.py // Full dataset preparation and training script
  - streamlit_app (1).ipynb // Streamlit app development in notebook
  - streamlit_app.py // Final Streamlit app code for deployment

---

Models Used for Final Prediction

Only two models were finally used in the Streamlit app:
- EfficientNet-B3_best
- EfficientNet-B4_best

These two models are loaded in the Streamlit app for prediction using a soft voting method.

---

How to Run the Streamlit App

1. Open a terminal inside the project folder.
2. Run this command:
// streamlit run streamlit_app.py //
3. A new browser window will open.
4. In the web app, you can:
   - Upload a fundus eye image
   - See the disease prediction
   - View the Grad-CAM heatmap for better understanding
   - Download a PDF report with the prediction result

---

Important Notes

- Only EfficientNet-B3 and EfficientNet-B4 models are used in the final version.
- Traditional ML models were trained separately but not used inside the Streamlit app.

---

Final Comments

This project helped me understand how to combine deep learning and web app development for real-world medical applications.
I have kept everything simple and organized to make it easy to check and run.
