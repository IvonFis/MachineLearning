# Data preprocessing
# we dont need to import the libraries, they are selected
# by default

# IMPORT THE DATA SET
dataset = read.csv('Data.csv')

# TAKE CARE OF MISSING DATA
dataset$Age = ifelse(is.na(dataset$Age),
                    ave(dataset$Age, FUN = function(x) mean(x, rm = TRUE)),
                    dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Age),
              ave(dataset$Salary, FUN = function(x) mean(x, rm = TRUE)),
              dataset$Salary)

# ENCODING CATEGORICAL DATA

dataset$Country = factor(dataset$Country,
                  levels = c('France', 'Spain', 'Germany'),
                  labels = c(1,2,3))

dataset$Purchased = factor(dataset$Purchased,
                  levels = c('No', 'Yes'),
                  labels = c(0,1))

# Splitting the data set
# install.packages('caTools')
library(caTools)

set.seed(0)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
