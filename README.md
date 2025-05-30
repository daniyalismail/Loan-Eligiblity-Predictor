# 🏦 Loan Eligibility Predictor

This project is a web application built with Streamlit that predicts loan eligibility based on various applicant details. It uses a Logistic Regression model trained on a loan approval dataset.

## 🌟 Features

* **Interactive Web Interface:** User-friendly interface built with Streamlit to input applicant details.
* **Real-time Prediction:** Instantly predicts whether a loan application is likely to be Approved (1) or Not Approved (0).
* **Model Performance Display:** Shows key performance metrics of the trained model in the sidebar:
    * Accuracy Score
    * Confusion Matrix
    * Classification Report
* **Data-Driven Insights:** Utilizes a dataset (`loan_approval_dataset.csv`) to train the prediction model.
* **User Guidance:** Includes clear instructions on how to use the application.
* **Error Handling:** Implements checks for data loading and input validation.
* **Clean UI:** Designed with a minimalistic and user-friendly aesthetic.

## 💻 Technology Stack

* **Python:** Core programming language.
* **Streamlit:** For creating the interactive web application.
* **Pandas:** For data manipulation and preprocessing.
* **Scikit-learn:**
    * `LogisticRegression` for the classification model.
    * `StandardScaler` for feature scaling.
    * `train_test_split` for splitting the dataset.
    * Metrics (`accuracy_score`, `confusion_matrix`, `classification_report`) for model evaluation.
* **Matplotlib & Seaborn:** For plotting the confusion matrix.

## 📊 Dataset

The model is trained using the `loan_approval_dataset.csv` file. The key features used from the dataset for prediction include:

* Number of Dependents
* Education Level (Graduate/Not Graduate)
* Self-Employment Status (Yes/No)
* Annual Income
* Loan Amount Requested
* Loan Term (in years)
* CIBIL Score (Credit Score)
* Value of Residential Assets
* Value of Commercial Assets
* Value of Luxury Assets
* Value of Bank Assets

The target variable is `loan_status` (Approved/Rejected).

## 🛠️ Setup and Installation

1.  **Clone the repository (if you've uploaded it to GitHub):**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```
    (Replace `your-username/your-repository-name` with your actual GitHub repo details)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    Activate the environment:
    * Windows: `venv\Scripts\activate`
    * macOS/Linux: `source venv/bin/activate`

3.  **Install the required Python libraries:**
    ```bash
    pip install streamlit pandas scikit-learn matplotlib seaborn
    ```

4.  **Dataset:**
    Ensure the `loan_approval_dataset.csv` file is present in the root directory of the project (the same directory as your Streamlit Python script, e.g., `app.py` or `loan_predictor_app.py`).

## ▶️ How to Run the Application

1.  Navigate to the project directory in your terminal.
2.  Make sure your virtual environment is activated (if you created one).
3.  Run the Streamlit application using the following command:
    ```bash
    streamlit run your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of your Python file, e.g., `loan_predictor_app.py`).

4.  The application will open in your default web browser.

## 📸 Screenshots (Optional)

*(You can add screenshots of your application here to give users a visual preview.)*

**Example:**
* Main input interface.
* Sidebar showing performance metrics.
* Prediction output (Approved/Not Approved).

To add images to your README:
1. Create a folder (e.g., `images` or `screenshots`) in your repository.
2. Add your screenshot images to this folder.
3. Link them in Markdown like this: `![Description of image](images/screenshot_name.png)`


## 🧑‍💻 Authors

This project was developed by:

* **M. Daniyal Ismail** (Roll No: 39)
* **Syed Maazin Imran** (Roll No: 23)

## 🚀 Potential Future Improvements

* **Advanced Models:** Experiment with other classification algorithms (e.g., Random Forest, Gradient Boosting, SVM) for potentially higher accuracy.
* **Hyperparameter Tuning:** Implement techniques like GridSearchCV or RandomizedSearchCV to find optimal hyperparameters for the chosen model.
* **Detailed EDA:** Include a section or a separate notebook for Exploratory Data Analysis of the dataset.
* **Feature Importance:** Display feature importance scores to understand which factors most influence loan approval.
* **User Accounts/History:** (More complex) Allow users to save their predictions or view a history.
* **Deployment:** Deploy the application to a cloud platform (e.g., Streamlit Sharing, Heroku, AWS, GCP) for public access.
* **More Robust Input Validation:** Add more specific validation rules for each input field (e.g., realistic ranges for income, loan amounts based on income).

---

*This README was generated based on the project requirements and code discussed.*
