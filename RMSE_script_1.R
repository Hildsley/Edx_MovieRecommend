#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

################ End of Creation of edx and validation sets

############### Start of Script
library(tidyverse)
library(caret)
library(lubridate)
library(anytime)

#   Create function which comptues the rmse 
#   between the actual and predicted ratings.
#   This function will be used to determine the
#   the relative accuracy of the predictions made by the model.

RMSE <- function(predicted_ratings, actual_ratings) {
  sqrt(mean((actual_ratings-predicted_ratings)^2))
  }

# Understanding the details of the dataset, like column names etc.

head(edx,n = 10)

# The amount of ratings of each user vary greatly 


# Plot showing the different average ratings for all the different movies

   edx %>% group_by(movieId) %>% summarise(avg_rating = sum(rating)/n()) %>% 
   ggplot(aes(x = avg_rating)) +
   geom_histogram(bins = 30, fill = "navy") + 
   ggtitle(label = "Average Rating For Different Movies") +
   xlab(label = "Average Movie Rating") + 
   ylab(label = "Movie Count")
 
   ###############################################
   ##### Start of Prediction algorithm ###########
   ###############################################   

##### Mean of  all movies calculation: prediction = mu (Step 1)

 mu <- mean(edx$rating) #Calculates the mean of entire dataset's ratings
 
 RMSE(mu,edx$rating) # RMSE of mu as a prediction only

##### Movie specific effects, distance from mu: prediction = mu + b_m (Step 2)
 
b_m <- edx %>% group_by(movieId) %>% summarise(b_m = sum(rating - mu)/n()) #Movie specific effect calculation

predicted_ratings_b_m <- edx %>% left_join(b_m , by = "movieId") %>% mutate(pred = mu + b_m) %>% .$pred # associated predictions after b_m was calculated

RMSE(predicted_ratings_b_m,edx$rating) # calculates RMSE

##### User specific effects, distance from mu: prediction = mu + b_u (Step 3)

edx %>% group_by(userId) %>% summarise(avg_rating = mean(rating)) %>% # Group by users to find the average of each user
  ggplot(aes(x = avg_rating)) +
  geom_histogram(bins = 30, fill = "navy") +
  ggtitle( label = "Average Rating For Individual Users") +
  xlab(label = "Average User Rating") +
  ylab(label = "User Count")   # Creates a histogram plot which shows the frequency of users' average ratings

b_u_seperate <- edx %>% group_by(userId) %>% summarise(b_u = mean(rating - mu)) # calculates user specific effects

predicted_ratings_b_u <- edx %>% left_join(b_u_seperate, by = "userId") %>% mutate(pred = mu + b_u) %>% .$pred # calculates associated predictions

RMSE(predicted_ratings_b_u,edx$rating)

##### User + Movie specific effects, distance from mu + b_m + b_u : prediction = mu + b_m + b_u (Step 4)

b_u <- edx %>% left_join(b_m , by = "movieId") %>% group_by(userId) %>% summarise(b_u = mean(rating - mu - b_m)) # calculate b_u

predicted_ratings_both <- edx %>% left_join(b_m, by = "movieId") %>% left_join(b_u , by = "userId") %>% mutate(pred = mu + b_m + b_u) %>% .$pred #prediction values

rmse_result_test <- RMSE(predicted_ratings_both,edx$rating)
rmse_result_test

max(predicted_ratings_both)  # Note that user's ratings may not be more than 5
min(predicted_ratings_both)  # or less than 0, therefore change those predicted ratings accordingly

for (x in 1:length(predicted_ratings_both)) {
  if (predicted_ratings_both[x] > 5) predicted_ratings_both[x] = 5   # changes ratings to a 5 for those above 5 predicted
  else if (predicted_ratings_both[x] < 0) predicted_ratings_both[x] = 0 # changes ratings to 0 for predicted less than 0 
}

rmse_result_test <- RMSE(predicted_ratings_both,edx$rating)
rmse_result_test


##### The above 4 steps did not include regularization of movies or users. Regularization 
##### ensures that movies that don't have many users' ratings' will have a lower predicted
##### rating due to more uncertainty. The next piece of code regularizes the predicted
##### ratings. The optimal penalty,lambda, will first be optimized for before running final 
##### prediction model.

lambda <- seq(0,3,0.1)  # sequence of penalty values,lambda, to test

rmses_reg_both <- sapply(lambda, function(l){  # will test each value of lambda and complete this function
  mu <- mean(edx$rating)
  b_m <- edx %>% group_by(movieId) %>% summarise(b_m = sum(rating - mu)/ (n()+l)) # regularizes movies first
   b_u <- edx %>% left_join(b_m, by = "movieId") %>% group_by(userId) %>% summarise(b_u = sum(rating - mu - b_m)/(n()+l)) # regularizes users second
predicted_ratings_reg_both <- edx %>% left_join(b_m, by = "movieId") %>% left_join(b_u , by = "userId") %>% mutate(pred = mu + b_m + b_u) %>% .$pred #prediction values
RMSE(predicted_ratings_reg_both,edx$rating)
}) # Will take a few minutes to run

plot(lambda,rmses_reg_both) # Plot the values to see the trend of lambda
lambda[which.min(rmses_reg_both)] # The value of lambda that minimizes the RMSE value

##### from the regularization steps above, lambda = 0.5 was defined as the penalty value that minimizes the RMSE
##### This penalty value will therefore be used in the prediction model

lambda_pred <- lambda[which.min(rmses_reg_both)] # This is the lambda used for the testing model, minimum of tested lambdas


b_m_reg <- edx %>% group_by(movieId) %>% summarise( b_m = sum(rating - mu)/ (n()+lambda_pred) ) # regularizes movies' effect
b_u_reg <- edx %>% left_join(b_m, by = "movieId") %>% group_by(userId) %>% summarise(b_u = sum(rating - mu - b_m)/(n()+lambda_pred)) # regularizes users' effect
predicted_ratings_reg_both <- edx %>% left_join(b_m, by = "movieId") %>% left_join(b_u , by = "userId") %>% mutate(pred = mu + b_m + b_u) %>% .$pred # prediction values

for (x in 1:length(predicted_ratings_reg_both)) {
  if (predicted_ratings_reg_both[x] > 5) predicted_ratings_reg_both[x] = 5   # changes ratings to a 5 for those above 5 predicted
  else if (predicted_ratings_reg_both[x] < 0) predicted_ratings_reg_both[x] = 0 # changes ratings to 0 for predicted less than 0 
}

RMSE(predicted_ratings_reg_both,edx$rating)

##### The Prediction Model's RMSE on the validation set

predict_val <- validation %>% left_join(b_m_reg , by = "movieId") %>% left_join(b_u_reg , by = "userId") %>% mutate(pred = mu + b_m + b_u) %>% .$pred

for (x in 1:length(predict_val)) {
  if (predict_val[x] > 5) predict_val[x] = 5
  else if (predict_val[x] < 0) predict_val[x] = 0
}

rmse_result_val <- RMSE(predict_val,validation$rating)
rmse_result_val 








