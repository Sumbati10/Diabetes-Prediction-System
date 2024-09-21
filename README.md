## Diabetes-Prediction-System
The Diabetes Prediction System is a web-based application that predicts the likelihood of an individual having diabetes based on various health-related features. The system utilizes a machine learning model trained on historical data to provide predictions for users.

# Diabetes classification 

The dataset contains 8 medical conditions features **(X)** : 

`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`,
       `BMI`, `DiabetesPedigreeFunction`, `Age`

`Outcome` is present a binary target **(y)** label 

>0 : Diabetes **False**

>1 : Diabetes **True**

dimension of diabetes data: **(768, 9)**
 * Dataset is small but well labeled. There are no null values present.
 * very suitable to supervised machine learning formulation.
 * This is a binary classification problem, where we have 2 classes in the target **(y)** (i.e.`df['Outcome']`) and the medical conditions can be used as the feature (**X**).
 
 
 # Machine-Learning Models
 
 I have used 8 different machine learning classifiers to diabetes classfication : 
 * K-Nearest Neighbors (kNN)
 * Logistic regression 
 * Decision Tree
 * Random Forest
 * Gradient Boosting
 * Support Vector Machine (SVM)
 * Neural Networks (Multi-level Perceptron : MLP)
 * XGBoost
 
 **Results are Shown below** 
 
  <img width="411" alt="image" src="https://github.com/user-attachments/assets/5ebdff28-e155-4328-a884-8e9f60b2ce54">

    
   # Conclusion 
   
   * **Logistic regression** and **Neural Netowrks** seems to provide the best performance based on **10-fold cross validation** of the dataset. **Logistic regression** achieves a higher *F1-score* as well, which is better metric for model evalution.
* From the confusion matricies, decision tree has the highest success in detecting the diabetes.
* **Feature selection** suggests the `Glucose` is the most crucial factor for the successful prediction of diabetes. 

## Getting Started
Clone the repository
```
git clone git clone https://github.com/Sumbati10/Diabetes-Prediction-System.git
```
Run the development server
```
cd Diabetes-Prediction-System

```
```
python app.py runserver
```
Access the application in your web browser at ```http://127.0.0.1:5000/```.

<img width="944" alt="demo" src="https://github.com/user-attachments/assets/4da0ac45-4645-433c-a1a1-1a1c9b5e14c6">


## Usage
- Open the application in your web browser.
- Fill in the required health-related information in the form.
- Click the "Submit" button to get the prediction result.
- View the prediction result on the page.


