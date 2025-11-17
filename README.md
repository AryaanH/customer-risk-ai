To run the API or scoring, first run the training notebook or train_model.py to generate models/final_risk_model.pkl

For this project I used Scikit-learn pipelines and tree based models such as Random Forest and Gradient Boosting for credit risk prediction.
I also built a small neural network in PyTorch to compare classical ML with a deep learning approach.
I evaluated all models using ROC AUC and decided to choose Random Forest as the production model because of its balance of performance and interpretability.”