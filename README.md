# Customer Risk Scoring & Early Warning AI System

A full end-to-end machine learning system that predicts customer default risk for financial services.

This project includes:
- Complete ML pipeline (EDA → feature engineering → model selection → PyTorch baseline)
- Production-ready FastAPI microservice running on Azure Web App for Containers
- Dockerized deployment with Linux/amd64 image
- Inference API exposed via /score endpoint
- Interactive Power BI dashboard (3 pages)

**Live API** https://aryaan-risk-api.azurewebsites.net/docs  

---

## 1. Problem Overview

Financial institutions need to identify customers who are likely to default so they can intervene early, reduce losses, and meet regulatory requirements.

---

## 2. Dataset & Features

Dataset: [Kaggle – Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)  
Target variable: `loan_status`  
- `0` = non-default  
- `1` = default  

### **Feature Groups**
**Customer**
- age, income, home ownership, employment length  

**Loan**
- amount, interest rate, intent, grade, percent of income  

**Credit History**
- previous defaults, credit history length  

### **Class Balance**
- 78% non-default  
- 22% default  

This imbalance makes **ROC-AUC** a better metric than accuracy.

---

## 3. Modelling Approach

### **Preprocessing Pipeline**
- Median imputation for numeric features  
- Most-frequent imputation for categorical features  
- Scaling numeric features  
- One-hot encoding categorical variables  

### **Models Compared**
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- Small neural network in **PyTorch**

### **Selected Model**
✔ **Random Forest**  
✔ **ROC-AUC ≈ 0.93**  
✔ Strong performance + robustness + explainability  

The pipeline and final model are saved as `.pkl` using `joblib`.

---

## 4. Inference System

The module `src/inference.py` loads the saved:
- Preprocessor  
- ML model  

And exposes two functions:
- `score_single_customer(dict)`  
- `score_batch(DataFrame)`  

### **Example Output**
```json
{
  "risk_score": 12.4,
  "risk_bucket": "Low"
}
```
---

## 5. API Service (FastAPI)

Two main endpoints:
**GET /health** for health check
**POST/score** to score a single cusetomer

### **Example Request**

```json
{
  "person_age": 35,
  "person_income": 60000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5.0,
  "loan_intent": "EDUCATION",
  "loan_grade": "B",
  "loan_amnt": 10000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.15,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 8
}
```

### **Example Response**

```json
{
    "risk_score": 12.4,
    "risk_bucket": "Low"
}
```
---

## 5. Running Locally


**Install env**
```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Run API**
```bash
uvicorn app.main:app --reload
```
Visit:
	•	http://127.0.0.1:8000/health
	•	http://127.0.0.1:8000/docs


## 👨‍💻 Author  

**Aryaan Hashim**  
*B.Eng. Software Engineering, Monash University*  
☁️ *Cloud & AI Engineer — Azure | AWS | Applied AI | Data Engineering*  
📍 *Melbourne, Australia*  

🔗 [LinkedIn](https://linkedin.com/in/aryaan-hashim) | [GitHub](https://github.com/AryaanH)


