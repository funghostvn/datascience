# 1. Installing essential packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

# 2. Data Wrangling
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
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

# 3. Loss function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# 4. Modeling
# Naive by average
mu <- mean(edx$rating) 
naive_rmse <- RMSE(edx$rating, mu)
rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)

# Add movie effect
movie_avgs <- edx %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))
predicted_ratings <- mu + validation %>% left_join(movie_avgs, by='movieId') %>% .$b_i
model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results, tibble(method="Movie Effect Model", RMSE = model_1_rmse ))

# Add user effect
user_avgs <- edx %>% left_join(movie_avgs, by='movieId') %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu - b_i))
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>% 
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>% .$pred
model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results, tibble(method="Movie + User Effects Model", RMSE = model_2_rmse ))

# Add genre effect
genre_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>% 
  left_join(user_avgs, by='userId') %>% 
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u))
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>% 
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>% .$pred
model_3_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results, tibble(method="Movie + User + Genre Effects Model", RMSE = model_3_rmse ))

# Add time stamp effect
week_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>% 
  left_join(user_avgs, by='userId') %>% 
  left_join(genre_avgs, by='genres') %>% 
  group_by(week) %>% 
  summarize(b_w = mean(rating - mu - b_i - b_u - b_g))
predicted_ratings <- validation %>% mutate(date = as_datetime(timestamp),week = round_date(date,"week")) %>%
  left_join(movie_avgs, by='movieId') %>% 
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(week_avgs, by='week') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_w) %>% .$pred
model_4_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results, tibble(method="Movie + User + Genre + Week Effects Model", RMSE = model_4_rmse ))

#rmse_results
#rmse_results %>% knitr::kable()


rmse_target <- 0.86490
rmse_results %>% mutate(RMSE_target = ifelse(RMSE < rmse_target, TRUE, FALSE ))

# 4. Modeling tuning
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-test_index,]
test <- edx[test_index,]

# Regularization
reglr_fit <- function(lambda, trainset, testset){
  mu <- mean(trainset$rating) 
  b_i <- trainset %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n()+lambda))
  b_u <- trainset %>% left_join(b_i, by="movieId") %>% 
    group_by(userId) %>% 
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  b_g <- trainset %>% left_join(b_i, by="movieId") %>% left_join(b_u, by='userId') %>%         
    group_by(genres) %>% 
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+lambda))    
  b_w <- trainset %>% left_join(b_i, by="movieId") %>% left_join(b_u, by='userId') %>% left_join(b_g, by='genres') %>% 
    group_by(week) %>% 
    summarize(b_w = sum(rating - b_g - b_i - b_u - mu)/(n()+lambda))
  
  predicted_ratings <- testset %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_w, by = "week") %>%   
    filter(!is.na(b_i), !is.na(b_u), !is.na(b_g), !is.na(b_w)) %>%
    mutate(pred = mu + b_i + b_u + b_g + b_w) %>%
    select(pred, rating)   
  return(RMSE(predicted_ratings$pred, predicted_ratings$rating))
}

lambdas <- seq(-0.5, 8, 0.1)
rmses <- sapply(lambdas, reglr_fit, trainset = train, testset = test)

tibble(Lambda = lambdas, RMSE = rmses) %>%
  ggplot(aes(x = Lambda, y = RMSE)) +
  geom_point()

# 5. Model with best parameter
lambda <- 5

mu <- mean(edx$rating) 
b_i <- edx %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n()+lambda))
b_u <- edx %>% left_join(b_i, by="movieId") %>% 
  group_by(userId) %>% 
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
b_g <- edx %>% left_join(b_i, by="movieId") %>% left_join(b_u, by='userId') %>%         
  group_by(genres) %>% 
  summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+lambda))    
b_w <- edx %>% left_join(b_i, by="movieId") %>% left_join(b_u, by='userId') %>% left_join(b_g, by='genres') %>% 
  group_by(week) %>% 
  summarize(b_w = sum(rating - b_g - b_i - b_u - mu)/(n()+lambda))

predicted_ratings <- validation %>% mutate(date = as_datetime(timestamp),week = round_date(date,"week")) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_w, by = "week") %>%   
  #filter(!is.na(b_i), !is.na(b_u), !is.na(b_g), !is.na(b_w)) %>%
  mutate(pred = mu + b_i + b_u + b_g + b_w) %>% .$pred
model_5_rmse <- RMSE(predicted_ratings, validation$rating)


# 6. Final results
rmse_results <- bind_rows(rmse_results, tibble(method="Movie + User + Genre + Week + Regularization Effects Model", RMSE = model_5_rmse ))
rmse_target <- 0.86490
rmse_results %>% mutate(RMSE_target_passing = ifelse(RMSE < rmse_target, TRUE, FALSE ))

