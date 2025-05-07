import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
try:
    df = pd.read_csv('Students_games_and_success_Rate.csv')
    print("Dataset loaded successfully!")
    print(df.head())
except FileNotFoundError:
    print("Error: The CSV file 'Students_games_and_success_Rate.csv' was not found. Please make sure the file is in the correct directory.")
    exit()

# 1. Data Cleaning: Handling Missing Values
print("\n--- 1. Data Cleaning: Handling Missing Values ---")
print("\nMissing values before handling:")
print(df.isnull().sum())

# For numerical columns, impute with the mean
numerical_cols = df.select_dtypes(include=np.number).columns
for col in numerical_cols:
    df[col].fillna(df[col].mean(), inplace=True)
print("\nMissing values after mean imputation for numerical columns:")
print(df.isnull().sum())

# For categorical columns, impute with the mode
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
print("\nMissing values after mode imputation for categorical columns:")
print(df.isnull().sum())

# 2. Data Integration: Not directly applicable with a single dataset

# 3. Data Reduction: Attribute Subset Selection
print("\n--- 3. Data Reduction: Attribute Subset Selection ---")
print("\nColumns in the original DataFrame:")
print(df.columns.tolist())

# Let's assume we want to focus on gaming habits and academic performance
# We'll select a subset of relevant columns
selected_columns = ['gender', 'grade', 'math_score', 'reading_score', 'writing_score',
                    'gaming_hours', 'favorite_genre', 'platform_preference']
df_reduced = df[selected_columns].copy()
print("\nDataFrame after selecting a subset of columns:")
print(df_reduced.head())

# 4. Data Transformation: Normalization
print("\n--- 4. Data Transformation: Normalization ---")
print("\nNumerical columns before normalization:")
print(df_reduced[['math_score', 'reading_score', 'writing_score', 'gaming_hours']].head())

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Select numerical columns for normalization
numerical_cols_to_normalize = ['math_score', 'reading_score', 'writing_score', 'gaming_hours']

# Apply normalization
df_reduced[numerical_cols_to_normalize] = scaler.fit_transform(df_reduced[numerical_cols_to_normalize])

print("\nNumerical columns after Min-Max normalization:")
print(df_reduced[['math_score', 'reading_score', 'writing_score', 'gaming_hours']].head())

# 5. Data Discretization: Histogram Analysis (Binning)
print("\n--- 5. Data Discretization: Histogram Analysis (Binning) ---")

# Let's discretize 'gaming_hours' based on histogram analysis (we'll define bins)
print("\nDistribution of 'gaming_hours' before discretization:")
print(df_reduced['gaming_hours'].hist(bins=5)) # You can adjust the number of bins
import matplotlib.pyplot as plt
plt.title('Histogram of Gaming Hours (Normalized)')
plt.xlabel('Normalized Gaming Hours')
plt.ylabel('Frequency')
plt.show()

# Define bins based on the distribution (you might need to adjust these)
bins = [0, 0.2, 0.5, 0.8, 1.0]
labels = ['Very Low', 'Low', 'Moderate', 'High']

# Apply discretization
df_reduced['gaming_hours_binned'] = pd.cut(df_reduced['gaming_hours'], bins=bins, labels=labels, right=False)

print("\n'gaming_hours' after discretization:")
print(df_reduced[['gaming_hours', 'gaming_hours_binned']].head())

print("\nData Preprocessing steps completed!")
print("\nProcessed DataFrame (first 5 rows):")
print(df_reduced.head())
