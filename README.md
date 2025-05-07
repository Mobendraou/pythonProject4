# Bank Customer Churn Prediction

This project predicts customer churn using a Decision Tree classifier. It includes exploratory data analysis, model training, hyperparameter tuning with GridSearchCV, and performance evaluation.

##  Features
- **Data Preprocessing**: Drops irrelevant columns and encodes categorical variables.
- **Exploratory Analysis**: Checks class balance and average balance of churned customers.
- **Model Training**: Decision Tree classifier with train-test split (stratified).
- **Hyperparameter Tuning**: Uses GridSearchCV to optimize `max_depth` and `min_samples_leaf`.
- **Performance Metrics**: Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
- **Visualization**: Decision Tree visualization and confusion matrix plot.

## Dependencies
- Python 3.x
- Libraries: numpy pandas matplotlib scikit-learn
