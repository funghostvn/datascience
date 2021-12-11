# loading libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(forecast)) install.packages("forecast", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")

# download data
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df_confirmed <- read.csv(url)
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
df_death <- read.csv(url)

# Reshape data
df_confirmed <- df_confirmed %>%
  gather(key = Date, value = TotalCase,-Province.State,-Country.Region,-Lat,-Long)
df_confirmed$Date <- as.Date(strptime(substr(df_confirmed$Date,2,length(df_confirmed$Date)-1),format="%m.%d.%y"))

df_death <- df_death %>%
  gather(key = Date, value = TotalDeath,-Province.State,-Country.Region,-Lat,-Long)
df_death$Date <- as.Date(strptime(substr(df_death$Date,2,length(df_death$Date)-1),format="%m.%d.%y"))

df <- left_join(df_confirmed,df_death,by=c("Country.Region","Province.State","Lat","Long","Date"))
df <- df %>% group_by(Country.Region) %>% arrange(Date) %>%
  mutate(NewCase = (TotalCase - lag(TotalCase,1))) %>%
  mutate(NewDeath = (TotalDeath - lag(TotalDeath,1)))

# First look in dataset
knitr::kable(tail(df, 6), caption = "The last records of dataset ")

# Total case by date
df %>% group_by (Date) %>% summarize(TotalCases = sum(TotalCase)) %>% 
  ggplot(aes(Date,TotalCases)) + 
  geom_point() +
  geom_smooth()

# Top Country overview
knitr::kable(df %>% group_by(Country.Region) %>% 
               summarise(current_case = max(TotalCase)) %>% 
               arrange(desc(current_case)) %>% 
               ungroup() %>%
               head(10), 
             caption = "Top country by case")

df %>% group_by(Country.Region) %>% 
  summarise(current_case = max(TotalCase)) %>% 
  arrange(desc(current_case)) %>% head(10) %>%
  ggplot(aes(reorder(Country.Region, current_case), current_case, fill = current_case)) +
  geom_bar(stat = "identity") +
  coord_flip()

knitr::kable(df %>% group_by(Country.Region) %>% 
               summarise(current_case = max(TotalCase)) %>% 
               arrange(current_case) %>% 
               ungroup() %>%
               head(10), 
             caption = "Bottom country by case")

# Top country trend
top_country <- df %>% group_by(Country.Region) %>% 
  summarise(current_case = max(TotalCase)) %>% 
  arrange(desc(current_case)) %>% 
  head(10)

inner_join(top_country, df %>% group_by(Date, Country.Region) %>% 
             summarise(TotalCase = sum(TotalCase),.groups = 'drop'),by = "Country.Region") %>% 
  ggplot(aes(Date,TotalCase,col = Country.Region)) + geom_line()

# Train. test split
date_index <- max(df$Date) %m-% months(2)
train <- df %>% filter(Date <= date_index) %>% group_by(Date) %>% summarize(TotalCases = sum(TotalCase))
test <- df %>% filter(Date > date_index) %>% group_by(Date) %>% summarize(TotalCases = sum(TotalCase))

df %>% group_by (Date) %>% summarize(TotalCases = sum(TotalCase)) %>%
  mutate(group = ifelse(Date <= date_index,'Train','Test')) %>%
  ggplot(aes(Date,TotalCases,col = group)) + geom_line()

# function to estimate RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Base line model
fit <- lm(TotalCases ~ Date, data = train)
y_hat <- predict(fit,test)
rmse <- RMSE(test$TotalCases,y_hat)
rmse_results <- tibble(method = "lm", RMSE = rmse)

# Caret model scan
set.seed(1, sample.kind="Rejection") 
models <- c("glm", "svmLinear", "knn", "rf")
fits <- lapply(models, function(model){
  print(model)
  train(TotalCases~Date, method = model, data =train)
  #y_hat <- predict(fit, newdata = test)
  #y_hat[is.na(y_hat)] = 0    
}) 

# save predict data
pred <- sapply(fits, function(object)
  predict(object, newdata = test))

pred[is.na(pred)] = 0

#Let show predict result on test set

caret_pred <- test %>% select(Date, TotalCases)
caret_pred <- cbind(caret_pred, pred)
colnames(caret_pred) <- c("Date","TotalCases",models)
knitr::kable(caret_pred %>% head(10), caption = "Top of result in caret model")

# Save RMSE result
rmse <- sapply(seq(1,length(models)), function(i) sqrt(mean((pred[,i] - test$TotalCases)^2)))
names(rmse) <- models

for (model in models){
  rmse_results <- bind_rows(rmse_results, tibble(method=model, RMSE = rmse[model] ))    
}

knitr::kable(rmse_results, caption = "RMSE of caret model")

# Forecast model scan
library(forecast)
forcast_model <- c("arima","ets","bats")
set.seed(2021)
forcast_pred <- sapply(forcast_model, function(model){
  y = train$TotalCases
  if (model == "arima"){
    fit <- auto.arima(y)
  } 
  else if (model == "ets"){
    fit <- ets(y)
  }
  else{
    fit <- tbats(y)        
  }
  y_hat <- forecast(fit, h=nrow(test)) %>% .$mean
  #rmse <- sqrt(mean((y_hat - test$TotalCases)^2))    
})
pred_result <- cbind(caret_pred,forcast_pred)

rmse_results <- sapply(3:ncol(pred_result), function(i){sqrt(mean((pred_result[,i] - pred_result$TotalCases)^2))})
names(rmse_results) <- colnames(pred_result)[3:ncol(pred_result)]
knitr::kable(tibble(Algorithm = names(rmse_results), RMSE=unlist(rmse_results)) %>% arrange(RMSE), caption = "RMSE includes forecast model")

# Model compare
pred_result %>%
  pivot_longer(!c('Date'), names_to = "algorithm", values_to = "y_hat") %>%
  ggplot(aes(Date,y_hat, col = algorithm)) + 
  geom_line() 

#Fine tuning for est
## Predict by Country
train_country <- df %>% filter(Date <= date_index) %>% 
  group_by(Country.Region,Date) %>%
  summarize(TotalCases = sum(TotalCase),.groups = 'drop')
#train_country

pred_country <- sapply(unique(train_country$Country.Region), function(country){
  # train with country have covid already
  y <- train_country %>% filter(Country.Region == country & TotalCases > 0) %>% pull(TotalCases)
  if (length(y) ==0){
    y_hat = 0
  } else if (length(y) < 100){
    y_hat = max(y)
  } else {
    fit <- ets(y)
    y_hat <- forecast(fit, h=nrow(test)) %>% .$mean
  }
  y_hat    
})

df_pred_country <- pred_country %>% as_tibble() %>% 
  mutate(Date = test$Date) %>%
  pivot_longer(!c('Date'), names_to = "Country.Region", values_to = "y_hat")

y_hat <- df_pred_country %>% group_by(Date) %>% summarize(y_hat = sum(y_hat)) %>% pull(y_hat)
RMSE(y_hat,test$TotalCases)

# Recheck model on train / test set
y_hat <- train %>% pull(TotalCases) %>% ets() %>% forecast(nrow(test))
rbind(train %>% mutate(Actual = TotalCases, Predict = TotalCases) %>% select(Date, Actual, Predict),
      test %>% mutate(Actual = TotalCases, Predict = y_hat$mean) %>% select(Date, Actual, Predict)) %>%
  ggplot(aes(x = Date)) +
  geom_line(aes(y = Predict, col = 'Predict')) +
  geom_line(aes(y = Actual, col = 'Actual'))+ 
  scale_color_manual(name = "", values = c("Actual" = "darkblue", "Predict" = "red"))

# Final prediction for next month
finalResult <- df %>% group_by(Date) %>% summarize(TotalCases = sum(TotalCase)) %>% 
  pull(TotalCases) %>%
  ets() %>% forecast(10) %>% summary() %>% as_tibble() %>%
  mutate(Date = max(df$Date))

finalResult$Date <- seq(max(df$Date)+1,max(df$Date)+10,by="days")
knitr::kable(finalResult %>% select('Date','Point Forecast','Lo 80','Hi 80','Lo 95','Hi 95'), caption = "Next 10 days  global total cases forecast ")

# 2 month prediction graph
df %>% group_by(Date) %>% summarize(TotalCases = sum(TotalCase)) %>% 
  pull(TotalCases) %>%
  ets() %>% forecast(60) %>% 
  autoplot()
