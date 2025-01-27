# INF6027 - Instructions on how to run the code

Overview

This project uses statistical and machine learning techniques to analyse the impact of the home team’s advantage in football matches. The analysis includes calculating predictive metrics, running a multinomial logistic regression model, evaluating the model, and visualizing key insights through various charts.

Requirements

Before running the code, ensure you have the following libraries installed:
	•	dplyr: Data manipulation
	•	zoo: Rolling averages
	•	tidyr: Data reshaping
	•	ggplot2: Data visualization
	•	corrplot: Correlation matrix visualization
	•	car: Variance Inflation Factor (VIF) calculations
	•	glmnet: Lasso regression
	•	caret: Confusion matrix and accuracy
	•	Metrics: Log-loss calculation
	•	reshape2: Data reshaping for visualizations

Install the required libraries by running:

install.packages(c("dplyr", "zoo", "tidyr", "ggplot2", "corrplot", "car", "glmnet", "caret", "Metrics", "reshape2"))

How to Run the Code
	1.	Clone or download the repository to your local machine.
	2.	Ensure you have the required dataset (results.csv) in the same directory as the script.
	3.	Open the script in R or RStudio.
	4.	Run the script section by section, following the detailed comments.

Steps in the Analysis

1. Data Preprocessing
	•	The dataset is loaded (results.csv) and the outcome variable (Win, Draw, Loss) is defined.
	•	Independent variables (predictors) are calculated:
	•	Average goals scored at home (avg_home_goals).
	•	Average goals conceded away (avg_away_goals_conceded).
	•	Recent form of both the home and away teams (recent_form_home, recent_form_away).
	•	Missing values are identified and handled by elimination.

2. Descriptive Statistics
	•	Frequency tables and outcome distribution are generated.
	•	Grouped statistics for predictors by match outcomes are calculated.
	•	Summary metrics are created for each team’s home and away performance.

3. Data Splitting
	•	The dataset is split into 70% training and 30% testing sets to evaluate model performance effectively.

4. Multinomial Logistic Regression Model
	•	A multinomial logistic regression model is trained using the training set.
	•	Predictors:
	•	avg_home_goals
	•	avg_away_goals_conceded
	•	recent_form_home
	•	recent_form_away

5. Model Evaluation
	•	The model’s performance is evaluated on the testing set:
	•	Confusion matrix
	•	Accuracy
	•	Log-loss

6. Visualizations
	•	Bar Chart: Distribution of match outcomes.
	•	Confusion Matrix Heatmap: Visual representation of the model’s predictions.
	•	Correlation Matrix: Relationships between predictors and the match outcome.
	•	Stacked Bar Chart: Comparison of home vs. away points for top FIFA-ranked countries.

7. Saving the Dataset
	•	The final dataset with computed predictors and model results is saved as:

home_advantage_model_prediction_with_visualisation_OHM.csv

Expected Outputs

Key Outputs
	1.	Model Metrics:
	•	Confusion matrix
	•	Accuracy
	•	Log-loss
	2.	Visualizations:
	•	Distribution of match outcomes.
	•	Heatmap of the confusion matrix.
	•	Correlation matrix for predictors and outcome.
	•	Stacked bar chart comparing home vs. away points.

How to Interpret the Results
	•	Correlation Matrix:
	•	Highlights which predictors (e.g., avg_home_goals, recent_form_home) correlate most strongly with the match outcome.
	•	Model Metrics:
	•	Accuracy: Indicates how often the model correctly predicts match outcomes.
	•	Log-loss: Evaluates the model’s probabilistic predictions; lower values indicate better performance.
	•	Stacked Bar Chart:
	•	Illustrates the distribution of home and away points for top FIFA-ranked teams, emphasizing home advantage trends.

Notes
	•	Ensure the results.csv dataset is preprocessed correctly and matches the format expected by the script.
	•	Adjust the seed (set.seed(123)) if you wish to replicate results with a different random data split.
