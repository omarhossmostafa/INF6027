# Created by Omar Mostafa - January 2025
# INF6027 - Introduction to Data Science

# -------------------------------------------------------------------------------

# Libraries Used
library(dplyr)      # For data manipulation
library(zoo)        # For rolling averages
library(tidyr)      # For reshaping data
library(ggplot2)    # For data visualization
library(corrplot)   # For correlation matrix visualization
library(car)        # For VIF calculation
library(glmnet)     # For lasso regression
library(caret)      # For confusion matrix and accuracy
library(Metrics)    # For log-loss calculation
library(reshape2)   # For reshaping data for visualizations

# -------------------------------------------------------------------------------
# Section 1: Defining Response Variable

data <- read.csv("results.csv") # Load the dataset

# Defining dependent variable "Match Outcome"
data$Outcome <- ifelse(data$home_score > data$away_score, "Win",
                       ifelse(data$home_score == data$away_score, "Draw", "Loss"))

# Convert Outcome to a factor (important for modeling)
data$Outcome <- as.factor(data$Outcome)

# -------------------------------------------------------------------------------
# Section 2: Calculating Predictors

# 2.1 Home Team's Average Goals Scored at Home
home_avg_goals <- data %>%
  group_by(home_team) %>%
  summarise(avg_home_goals = mean(home_score, na.rm = TRUE)) # Calculate average goals
data <- data %>%
  left_join(home_avg_goals, by = "home_team") # Add to the dataset

# 2.2 Away Team's Average Goals Conceded Away
away_avg_goals_conceded <- data %>%
  group_by(away_team) %>%
  summarise(avg_away_goals_conceded = mean(home_score, na.rm = TRUE)) 
data <- data %>%
  left_join(away_avg_goals_conceded, by = "away_team") 

# 2.3 Recent Form of Home Team (Last 5 Matches)
data <- data %>%
  mutate(home_points = ifelse(home_score > away_score, 3, # 3 points for win
                              ifelse(home_score == away_score, 1, 0))) # 1 point for draw
recent_form_home <- data %>%
  group_by(home_team) %>%
  arrange(date) %>%
  mutate(recent_form_home = rollapply(home_points, 5, mean, fill = NA, align = "right")) # Rolling average
data <- left_join(data, recent_form_home %>% select(home_team, date, recent_form_home), 
                  by = c("home_team", "date")) 

# 2.4 Recent Form of Away Team (Last 5 Matches)
data <- data %>%
  mutate(away_points = ifelse(away_score > home_score, 3, 
                              ifelse(away_score == home_score, 1, 0))) 
recent_form_away <- data %>%
  group_by(away_team) %>%
  arrange(date) %>%
  mutate(recent_form_away = rollapply(away_points, 5, mean, fill = NA, align = "right")) 
data <- left_join(data, recent_form_away %>% select(away_team, date, recent_form_away), 
                  by = c("away_team", "date"))

# Missing Values' Analysis
print("Missing Values by Column:")
missing_values <- colSums(is.na(data))
print(missing_values)

# -------------------------------------------------------------------------------
# Section 3: Descriptive Statistics

# 3.1 Frequency Table
print("Frequency Table:")
frequency_table <- table(data$Outcome)
print(frequency_table)

# 3.2 Outcome Distribution
print("Outcome Distribution:")
outcome_distribution <- data %>%
  group_by(Outcome) %>%
  summarise(count = n(), percentage = n() / nrow(data) * 100)
print(outcome_distribution)

# 3.3 Grouped Statistics by Outcome
print("Grouped Statistics by Outcome:")
grouped_stats <- data %>%
  group_by(Outcome) %>%
  summarise(avg_home_goals_mean = mean(avg_home_goals, na.rm = TRUE),
            avg_away_goals_conceded_mean = mean(avg_away_goals_conceded, na.rm = TRUE),
            recent_form_home_mean = mean(recent_form_home, na.rm = TRUE),
            recent_form_away_mean = mean(recent_form_away, na.rm = TRUE))
print(grouped_stats)

# 3.4 Home Advantage in Each Country
# Calculate home and away points for each team
team_points <- data %>%
  mutate(
    home_points = ifelse(home_score > away_score, 3, ifelse(home_score == away_score, 1, 0)),  # Points at home
    away_points = ifelse(away_score > home_score, 3, ifelse(away_score == home_score, 1, 0))   # Points away
  ) %>%
  group_by(home_team) %>%
  summarise(
    games_played = n(),  # Total games played
    home_points_won = sum(home_points, na.rm = TRUE),  # Total points at home
    away_points_won = sum(away_points, na.rm = TRUE),  # Total points away
    total_points = home_points_won + away_points_won,  # Total points overall
    percent_home_points = (home_points_won / total_points) * 100  # Percentage at home
  ) %>%
  arrange(desc(percent_home_points))  # Sort by percentage of points at home

# Filter for teams with at least 50 games played
team_points_filtered <- team_points %>%
  filter(games_played >= 50)

print(team_points_filtered)

# -------------------------------------------------------------------------------
# Section 4: Data Cleaning & Preprocessing

# 4.1 Handling missing values by elimination
data <- data[complete.cases(data[, c("avg_home_goals", "avg_away_goals_conceded", 
                                     "recent_form_home", "recent_form_away")]), ]

# 4.2 VIF Calculations
# For average home goals
vif_model_1 <- lm(avg_home_goals ~ avg_away_goals_conceded + recent_form_home + recent_form_away, data = data)
vif_values_1 <- vif(vif_model_1)
print(vif_values_1)

# For average goals conceded away from home
vif_model_2 <- lm(avg_away_goals_conceded ~ avg_home_goals + recent_form_home + recent_form_away, data = data)
vif_values_2 <- vif(vif_model_2)
print(vif_values_2)

# For recent form home
vif_model_3 <- lm(recent_form_home ~ avg_home_goals + avg_away_goals_conceded + recent_form_away, data = data)
vif_values_3 <- vif(vif_model_3)
print(vif_values_3)

# For recent form away
vif_model_4 <- lm(recent_form_away ~ avg_home_goals + avg_away_goals_conceded + recent_form_home, data = data)
vif_values_4 <- vif(vif_model_4)
print(vif_values_4)

# 4.3 Regularisation with Lasso Regression

# Prepare the data for lasso regression
X <- as.matrix(data[, c("avg_home_goals", "avg_away_goals_conceded","recent_form_home", "recent_form_away")]) # Select predictors after VIF
y <- as.factor(data$Outcome) # Response variable

# Fit a lasso regression model
lasso_model <- cv.glmnet(X, y, family = "multinomial", alpha = 1)
coef(lasso_model, s = "lambda.min")

# -------------------------------------------------------------------------------
# Section 5: Multinomial Logistic Regression Model

# 5.1: Data Splitting 

# Split the data into 70% training and 30% testing
set.seed(123) # Set a random seed for reproducibility
train_index <- createDataPartition(data$Outcome, p = 0.7, list = FALSE)
train_data <- data[train_index, ]  # Training set
test_data <- data[-train_index, ]  # Testing set

# 5.2 Run Multinomial Logistic Regression Model
final_model <- multinom(Outcome ~ avg_home_goals + avg_away_goals_conceded + recent_form_home + recent_form_away, data = data)
summary(final_model)

# -------------------------------------------------------------------------------
# Section 6: Evaluation Metrics

# Predict outcomes on the test set
predicted <- predict(final_model, newdata = test_data)

# 6.1 Confusion Matrix
confusion_matrix <- confusionMatrix(as.factor(predicted), as.factor(test_data$Outcome))
  print(confusion_matrix)

# 6.2 Calculate Accuracy
accuracy <- mean(predicted == test_data$Outcome)
  print(paste("Accuracy: ", round(accuracy, 2)))

# 6.3 Log-Loss
probabilities <- predict(final_model, newdata = test_data, type = "probs")
  log_loss <- logLoss(actual = as.numeric(test_data$Outcome), predicted = probabilities)
    print(paste("Log-Loss: ", round(log_loss, 4)))
# -------------------------------------------------------------------------------
# Section 7: Visualisations

# 7.1 Bar Chart for Outcome Distribution
ggplot(data, aes(x = Outcome)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Match Outcomes when Playing at Home", x = "Outcome", y = "Count") +
  theme_minimal()

# 7.2 Confusion Matrix Heatmap 
confusion <- table(Predicted = predicted, Actual = test_data$Outcome)
  confusion_melted <- melt(confusion)

# Plot the heatmap
ggplot(confusion_melted, aes(x = Actual, y = Predicted, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), color = "white") +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(
    title = "Confusion Matrix Heatmap",
    x = "Actual Outcome",
    y = "Predicted Outcome"
  ) +
  theme_minimal()

# 7.3 Correlation Matrix
data$OutcomeNumeric <- as.numeric(data$Outcome) # Convert Outcome to numeric (1: Win, 2: Draw, 3: Loss)
cor_data <- data %>%
  select(avg_home_goals, avg_away_goals_conceded, recent_form_home, recent_form_away, OutcomeNumeric)
  cor_matrix <- cor(cor_data, use = "complete.obs")
    colnames(cor_matrix) <- c("Avg Goals Home", "Avg Goals Conceded Away", 
                          "Recent Form Home", "Recent Form Away", "Match Outcome (Numeric)")
    rownames(cor_matrix) <- c("Avg Goals Home", "Avg Goals Conceded Away", 
                          "Recent Form Home", "Recent Form Away", "Match Outcome (Numeric)")

corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45, addCoef.col = "black",
         title = "Correlation Between Variables and Match Outcome")


# 7.4 Stacked Bar Chart - Home vs Away Points For Highest FIFA Ranked Countries

selected_countries <- c("Brazil", "Argentina", "Spain", "France", "Italy", "England", 
                        "Uruguay", "Netherlands", "Egypt", "Germany")

# Filter the top_countries dataset to include only the selected countries
filtered_data <- team_points_filtered %>%
  filter(home_team %in% selected_countries) %>%
  select(home_team, home_points_won, away_points_won) %>%
  pivot_longer(cols = c(home_points_won, away_points_won), names_to = "Type", values_to = "Points")
  
# Create the stacked bar chart
ggplot(filtered_data, aes(x = reorder(home_team, -Points), y = Points, fill = Type)) +
  geom_bar(stat = "identity") +
  labs(
    title = "Home vs. Away Points for Top FIFA Ranked Countries",
    x = "Country",
    y = "Points",
    fill = "Points Type"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# -------------------------------------------------------------------------------
# Section 8: Save the Updated Dataset

write.csv(data, "home_advantage_model_prediction_with_visualisation_OHM.csv", row.names = FALSE)
