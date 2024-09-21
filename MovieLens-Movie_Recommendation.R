##########################################################
# MovieLens Movie Recommendation System: 
# HarvardX Data Science Capstone Project
# Author: Thant Thiha
# Date: 2024-09-21
##########################################################

## Introduction
# In this project, I aim to build a movie recommendation system using the MovieLens 10M dataset.
# The goal is to predict movie ratings based on user preferences using collaborative filtering techniques.
# The performance of the model will be evaluated using Root Mean Squared Error (RMSE).

# Key steps:
# 1. Data Understanding and Preparation,
# 2. Exploratory Data Analysis,
# 3. Model Development,
# 4. Final Model Evaluation.

## 1. Data Preparation
# This section includes downloading the MovieLens dataset, merging movie and rating information, 
# and splitting the data into training (edx) and evaluation (final_holdout_test) sets.

##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: This process could take a couple of minutes
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE), stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId), movieId = as.integer(movieId), rating = as.numeric(rating), timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE), stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Ensure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

## 2. Exploratory Data Analysis (EDA)
# In this section, let's explore the data to understand the distribution of ratings and other key characteristics.

# Check the structure of the edx dataset
str(edx)

# Check a summary of the dataset
summary(edx)

# Count unique users and movies
n_users <- n_distinct(edx$userId)
cat("No of unique users:", n_users, "\n")
n_movies <- n_distinct(edx$movieId)
cat("No of unique movies:", n_movies, "\n")

# Distribution of ratings
edx %>%
  ggplot(aes(rating)) + 
  geom_histogram(bins = 30, color = "black", fill = "grey") + 
  ggtitle("Distribution of Movie Ratings")

## 3. Model Development
# This section involves building multiple models to progressively improve the prediction accuracy.

### 3.1. Baseline Model - Global Average
# This simple model predicts the average rating for all movies.

# Calculate the global average rating
mu <- mean(edx$rating)

# Calculate RMSE for the global average model
rmse_global_avg <- sqrt(mean((edx$rating - mu)^2))
cat("RMSE Global Average Rating:", rmse_global_avg, "\n")

### 3.2. Movie Effect Model
# This model adjusts predictions by incorporating the specific effect of each movie.

# Movie Effect Model: deviation of each movie's average rating from the global average
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Predict ratings using movie effects
validation_preds <- edx %>%
  left_join(movie_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

# RMSE calculation
rmse_movie_effect <- RMSE(edx$rating, validation_preds)
cat("RMSE of Movie Effect Model:", rmse_movie_effect, "\n")

### 3.3. Regularized User and Movie Effect Model
# To avoid overfitting, we apply regularization to movie and user effects.

# Define a regularization parameter
lambda <- 5

# Regularize movie effects
movie_avgs_reg <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + lambda))

# Regularize user effects
user_avgs <- edx %>%
  left_join(movie_avgs_reg, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))

# Predict ratings with regularized movie and user effects
validation_preds <- edx %>%
  left_join(movie_avgs_reg, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Calculate RMSE for the Movie and User effect model
rmse_final_model <- RMSE(edx$rating, validation_preds)
cat("RMSE of Regularized Movie + User Effect Model:", rmse_final_model, "\n")

## 4. Final Model Evaluation
# Evaluate the selected model on the final_holdout_test set to obtain the final RMSE.

# Predict on the final_holdout_test set using the best model
final_predictions <- final_holdout_test %>%
  left_join(movie_avgs_reg, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Calculate RMSE on the final holdout test set
rmse_holdout <- RMSE(final_holdout_test$rating, final_predictions)
cat("Final RMSE on Holdout Test Set:", rmse_holdout, "\n")

## Summary of Results
# RMSE of Global Average Model: 1.0603
# RMSE of Movie Effect Model: 0.9423
# RMSE of Regularized Movie + User Effect Model: 0.8570
# Final RMSE on Holdout Test Set: 0.8648

## Conclusion
# This analysis highlights the importance of considering both movie and user effects in predicting movie ratings.
# Despite achieving solid results, the analysis has limitations, including reliance on historical data 
# and potential outlier influence. 
# Future work should explore advanced techniques and incorporate additional features 
# to enhance model performance and insights.
