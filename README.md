# NutriClass: Food Classification Using Nutritional Data

## Project Overview 📊

This project develops a machine learning model to classify food items into distinct categories based on their nutritional attributes. By leveraging a dataset containing information on calories, protein, carbohydrates, and fat, we aim to build a robust classification system and gain insights into the nutritional profiles that define different food types.

### **Business Use Cases:**
* **Smart Dietary Applications:** Can be integrated into apps to recommend balanced meals.
* **Health Monitoring Tools:** Assists nutritionists in classifying food for personalized diet plans.
* **Food Logging Systems:** Automatically categorizes user-entered food data for easier tracking.

---

## Methodology 🧪

The project followed a standard machine learning pipeline:

### **1. Data Understanding and Preprocessing**
* **Dataset:** We used a tabular dataset with 8,081 rows and 11 columns.
* **Cleaning:** Missing values were handled by **imputing them with the median**.
* **Feature Engineering:** Features were scaled using `StandardScaler` and dimensionality was reduced with **Principal Component Analysis (PCA)**.
* **Label Encoding:** Categorical labels were converted into numerical format using `LabelEncoder`.

### **2. Model Selection and Training**
We trained and evaluated seven different classification models to find the best performer for this task:
* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors
* Support Vector Machine
* XGBoost
* Gradient Boosting Classifier

### **3. Evaluation Metrics**
Model performance was evaluated using standard metrics, including:
* **Accuracy**
* **Precision, Recall, and F1-score**
* **Confusion Matrix**

---

## Results and Key Insights 📈

The **Random Forest Classifier** emerged as the best-performing model, achieving an accuracy of **88.62%** and a high F1-score of **0.8864**.

The high accuracy suggests that nutritional data is an excellent predictor for food classification. The confusion matrix further highlighted the model's ability to correctly distinguish between different food categories.

---

## Project Deliverables

* **Source Code:** The Jupyter Notebook file (`nutriclass_report.ipynb`) containing all the code.
* **Report:** This README file serves as a comprehensive report, summarizing the project from start to finish.
* **Visualizations:** Key plots are included directly in the notebook to visualize data distribution, correlations, and model performance.

---

## Technical Stack 💻
* **Languages:** Python
* **Libraries:** Scikit-learn, Pandas, Matplotlib/Seaborn

---

This project taught me a lot about data preprocessing, feature engineering (using PCA), and model evaluation. The final model could be a great tool for smart dietary apps and health monitoring systems.

Check out the full project code and report on GitHub: [Link to your GitHub repository]

#MachineLearning #DataScience #Python #NutriClass #FoodClassification #ScikitLearn #GitHub #Project #Analytics #AI #Innovation
