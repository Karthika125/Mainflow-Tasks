import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Generate Sample Datasets
def generate_sample_datasets():
    global_superstore_data = {
        'Sales': np.random.randint(100, 5000, 100),
        'Profit': np.random.randint(-500, 2000, 100),
        'Region': np.random.choice(['East', 'West', 'North', 'South'], 100),
        'Category': np.random.choice(['Furniture', 'Technology', 'Office Supplies'], 100)
    }
    
    sales_data = {
        'Product': np.random.choice(['Laptop', 'Printer', 'Desk', 'Chair'], 100),
        'Region': np.random.choice(['East', 'West', 'North', 'South'], 100),
        'Sales': np.random.randint(100, 5000, 100),
        'Profit': np.random.randint(-500, 2000, 100),
        'Discount': np.random.uniform(0, 0.5, 100),
        'Category': np.random.choice(['Furniture', 'Technology', 'Office Supplies'], 100),
        'Date': pd.date_range(start='1/1/2023', periods=100)
    }
    
    global_superstore_df = pd.DataFrame(global_superstore_data)
    sales_df = pd.DataFrame(sales_data)
    
    global_superstore_df.to_csv("global_superstore.csv", index=False)
    sales_df.to_csv("sales_data.csv", index=False)
    print("Sample datasets generated successfully!")

# Load Dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df


def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Statistical Summary
def statistical_summary(df):
    return df.describe()

# Correlation Analysis
def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', fmt='.2f')

    plt.title('Feature Correlation Heatmap')
    plt.show()

# Data Visualization
def plot_histograms(df, columns):
    df[columns].hist(figsize=(10, 6), bins=20)
    plt.show()

def plot_boxplots(df, columns):
    plt.figure(figsize=(10, 6))
    df[columns].plot(kind='box', subplots=True, layout=(1, len(columns)), figsize=(12, 6))
    plt.show()

def plot_scatter(df, x_col, y_col):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[x_col], y=df[y_col])
    plt.title(f'{y_col} vs {x_col}')
    plt.show()

# Linear Regression Model
def train_regression_model(df, features, target):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f'R2 Score: {r2}')
    print(f'Mean Squared Error: {mse}')
    return model

# Main Execution
if __name__ == "__main__":
    generate_sample_datasets()
    
    file_path = "sales_data.csv"
    df = load_data(file_path)
    df = clean_data(df)
    df = handle_outliers(df, 'Sales')
    
    print("Statistical Summary:")
    print(statistical_summary(df))
    
    plot_correlation_heatmap(df)
    plot_histograms(df, ['Sales', 'Profit', 'Discount'])
    plot_boxplots(df, ['Sales', 'Profit', 'Discount'])
    plot_scatter(df, 'Profit', 'Sales')
    
    model = train_regression_model(df, ['Profit', 'Discount'], 'Sales')
