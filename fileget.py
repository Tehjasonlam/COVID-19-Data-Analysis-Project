import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the full dataset
df = pd.read_csv("owid_covid_data.csv")

# Filter for rows with non-null ICU patients
df_filtered = df[['total_cases', 'total_deaths', 'icu_patients',
                  'population_density', 'median_age', 'hospital_beds_per_thousand',
                  'gdp_per_capita', 'stringency_index']].dropna()

print(f"Rows with complete data: {len(df_filtered)}")

# Optional: Remove extreme outliers in total_cases and total_deaths
# (e.g. remove rows with values above the 99th percentile)
case_threshold = df_filtered['total_cases'].quantile(0.99)
death_threshold = df_filtered['total_deaths'].quantile(0.99)
df_filtered = df_filtered[
    (df_filtered['total_cases'] <= case_threshold) &
    (df_filtered['total_deaths'] <= death_threshold)
]

print(f"Rows after removing outliers: {len(df_filtered)}")

# Log transformation to reduce skewness
df_filtered['log_total_cases'] = np.log1p(df_filtered['total_cases'])
df_filtered['log_total_deaths'] = np.log1p(df_filtered['total_deaths'])

# Define features
X = df_filtered[['log_total_cases', 'log_total_deaths',
                 'population_density', 'median_age',
                 'hospital_beds_per_thousand', 'gdp_per_capita',
                 'stringency_index']]
y = df_filtered['icu_patients']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.3f}")

# Feature Importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importances:")
print(feature_importances.sort_values(ascending=False))
