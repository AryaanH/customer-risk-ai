from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    person_age: int
    person_income: int
    person_home_ownership: str
    person_emp_length: float | None = None
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float | None = None
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

class RiskResponse(BaseModel):
    risk_score: float
    risk_bucket: str