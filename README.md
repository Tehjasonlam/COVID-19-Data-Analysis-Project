There is different from the one my group submitted because they didnt take my work and just did everything last min and turned it in. I also found out last min also. Also msged the Ta.

# COSC 3337 Group Project: EDA on COVID-19 Dataset

## Introduction

This project explores the COVID-19 dataset provided by Our World in Data (OWID). The dataset (`owid_covid_data.csv`) includes information on COVID-19 cases, deaths, testing rates, hospitalization metrics, and more across various countries and regions. Through exploratory data analysis (EDA) and machine learning techniques, we analyzed the pandemic's impact and derived meaningful insights.

The project addresses multiple tasks ranging from identifying the most affected countries to predicting healthcare needs using regression models.

## Dataset

The dataset used in this project is `owid_covid_data.csv`. This dataset is included in the repository to ensure reproducibility.

## Project Structure

The project includes the following files:
- **`owid_covid_data.csv`**: The main dataset used for analysis.
- **`group_project_code.ipynb`**: Jupyter Notebook containing all the code for data preprocessing, analysis, visualizations, and machine learning.
- **`Group Project.pdf`**: A detailed project report explaining the objectives, methodology, findings, and conclusions.
- **`Group Project.docx`**: An alternate DOCX version of the project report.hospitalization_data.csv
- **`hospitalization_data.csv`**: Output of the hospitalization_data task 6.


## Tasks

### Task 1: Impact of COVID-19 on Countries
- **Objective**: Identify the top 10 countries most affected by COVID-19 based on total cases and deaths.
- **Analysis**: Calculate total cases and deaths per country. Visualize the results using bar plots.

### Task 2: Testing Rates Across Countries
- **Objective**: Evaluate testing efforts using the ratio of total tests to population for the top 10 most populated countries.
- **Analysis**: Compute testing rates and visualize discrepancies across the most populated nations using horizontal bar charts.

### Task 3: Comparing Rates of Increase Between Two Countries
- **Objective**: Compare the daily increase in COVID-19 cases between the United States and China for 2023.
- **Analysis**: Visualize trends over time using line plots.

### Task 4: Outlier Detection
- **Objective**: Detect and analyze outliers in key metrics such as `new_cases`, `new_deaths`, `icu_patients`, and `hosp_patients`.
- **Analysis**: Use statistical methods (e.g., IQR, Z-scores) and visualize outliers using box plots.

### Task 5: Regression Analysis
- **Objective**: Predict ICU admissions (`icu_patients`) based on features such as `total_cases` and `total_deaths`.
- **Analysis**: Train a linear regression model, evaluate its performance using metrics like R-squared and mean squared error, and visualize predictions.

### Hospitalization Metrics
- **Objective**: Calculate and categorize hospitalization needs for each country.
- **Analysis**: Compute `hospitalization_rate` as the percentage of cases requiring hospitalization. Categorize needs into "Very Low," "Low," "Medium," and "High" based on thresholds.

## Libraries Used
The project uses the following Python libraries:
- `pandas`: Data manipulation and analysis.
- `matplotlib`: Data visualization.
- `seaborn`: Enhanced data visualization.
- `scikit-learn`: Machine learning for regression analysis.

## How to Run the Code

1. **Import Libraries and Load the Dataset**:
   Ensure all required libraries are installed. Load the dataset using `pandas`.

2. **Filter and Preprocess Data**:
   - Remove non-country rows (e.g., regions marked with `OWID_` in `iso_code`).
   - Convert columns to appropriate data types (e.g., `date` to datetime).

3. **Run Tasks**:
   - Task 1: Analyze the most affected countries.
   - Task 2: Evaluate testing rates for the 10 most populated countries.
   - Task 3: Compare daily increase rates between selected countries.
   - Task 4: Detect and analyze outliers in key metrics.
   - Task 5: Train a regression model to predict healthcare needs.

4. **View Results**:
   - Bar plots for affected countries.
   - Horizontal bar charts for testing rates.
   - Line plots comparing case increases between two countries.
   - Box plots highlighting outliers.
   - Scatter plots for actual vs. predicted ICU admissions.

## Results

The provided code generates figures and statistics corresponding to the analysis in the project report. Key insights include:
- Identification of the most affected countries based on cases and deaths.
- Variations in testing rates among the most populated nations.
- Comparative trends in COVID-19 case increases between the United States and China.
- Outlier detection in key metrics such as new cases and ICU admissions.
- Prediction of ICU admissions using regression models.

## Contributors

- **Tri Lam (1916079)**