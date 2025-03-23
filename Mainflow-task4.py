import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("Housing.csv")

# Inspect Dataset
print("Dataset Overview:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())

# Handle Missing Values (Drop or Impute)
df.dropna(inplace=True)

# Analyze Numerical Features
sns.pairplot(df, diag_kind='kde')
plt.show()

# Identify and Remove Outliers (Using IQR Method)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = remove_outliers(df, 'area')
df = remove_outliers(df, 'price')

# Separate Features and Target Variable
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning']]
y = df['price']

# Data Preprocessing (Scaling & Encoding)
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['area', 'bedrooms', 'bathrooms', 'stories']),
    ('cat', OneHotEncoder(drop='first'), ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning'])
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Model Pipeline
model = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate Model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Root Mean Square Error: {rmse}")
print(f"R² Score: {r2}")

# Feature Importance Analysis
feature_names = model.named_steps['preprocess'].get_feature_names_out()
coefs = model.named_steps['regressor'].coef_
feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
print("\nFeature Importance:\n", feature_importance.sort_values(by='Coefficient', ascending=False))
