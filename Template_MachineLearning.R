# Basic template for Data Preprocessing for Machine Learning

# We import the file
dataset = read.csv('filename.csv')
# OPTIONAL: Take a subset of the data and the indexes of the columns for build the model
# for example column 2 and 3
# dasatet = dataset[, 2:3]

# Splitting the data set
# install.packages('caTools')
library(caTools)

set.seed(230)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# OPTIONAL: Feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])
