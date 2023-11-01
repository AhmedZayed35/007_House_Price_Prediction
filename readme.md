# House Price Prediction

This project aims to build a machine learning model to predict house prices based on various features. The dataset used for training and testing the model is provided in the 'data/housing.csv' file.

## Data Source

The [California Housing Prices dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices) used in this project was obtained from Kaggle. You can find the dataset and related information on Kaggle's website.

## Getting Started

### Prerequisites

To run the code, you need to have the following prerequisites:

- Python
- Pandas
- Matplotlib
- Scikit-learn
- seaborn

### Installation

Install the required packages after navigating to the project directory

   ```pip install -r requirements.txt```

### Code Execution

1. Run the Jupyter Notebook houss_prediction.ipynb to train, evaluate, and test the machine learning models.

2. The notebook includes data exploration, preprocessing, model building (Linear Regression and Random Forest Regressor), and hyperparameter tuning.

3. The code also demonstrates feature engineering and data visualization.

### Preprocessing and Feature Engineering

- The data is preprocessed to handle missing values, apply logarithmic transformations, and one-hot encode categorical features.
- Feature engineering is performed to create new features such as the bedroom ratio, population per household, and rooms per household.

### Model Building

- Two machine learning models are used for house price prediction: Linear Regression and Random Forest Regressor.
- Hyperparameter tuning is applied to the Random Forest Regressor to optimize its performance.

### Evaluation

- The models are evaluated based on their performance on a test dataset.
- Evaluation metrics include R-squared (coefficient of determination) and mean squared error.
