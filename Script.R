#loading "The Rolling Stones" dataset
dataset <- read.csv('stones_analysis.csv',stringsAsFactors = FALSE,check.names = FALSE)

#examining the structure of dataset
str(dataset)
#examining the summary of dataset
summary(dataset)       

#transforming dataset
source('Utility.R')
dataset <- transormed.dataframe(dataset = dataset)

#checking for NA values
all(complete.cases(dataset))

# creating outcome variable 'OnChart'
# Create a logical matrix indicating where the value is "No"
# Date variable will be excluded, since it doesn't make sense to transform it
# to numeric variable
dataset$Date <- NULL

first.chart <- which(colnames(dataset) == 'British charts')
last.chart <- which(colnames(dataset) == 'POL')
no.matrix <- dataset[, first.chart : last.chart] == "No"
no.matrix <- rowSums(!no.matrix) > 0

dataset$OnChart <- as.factor(ifelse(no.matrix == 'TRUE', "Yes","No"))
#eliminating variables that were used for creating outcome variable
dataset <- dataset[,-c(first.chart:last.chart)]

table(dataset$OnChart)
prop.table(table(dataset$OnChart))
# 226 in dataset have not been on any charts, whereas 71 have been
# Since there is a noticeable difference between number of observations in each class, there may
# be a problem when using kNN algorithm, since it is sensitive to data participation disbalance

# for kNN analysis all predictor variables are numerical and should be standardizes
length(unique(dataset$Title))
#too many values to be transformed to numeric
dataset$Title <- NULL
length(unique(dataset$`Album name`))
# Album name attribute will be excluded from model because it doesn't make sense to 
# transform it to numeric values
dataset$`Album name` <- NULL
# 'Record Label', 'Songwriter s' and 'Lead vocals' variables will also be excluded since it doesn't make sense to transform them to numeric variable
dataset$`Record Label` <- NULL
dataset$`Songwriter s ` <- NULL
dataset$`Lead vocal s ` <- NULL

length(unique(dataset$`Album type`))
# 'Album type' variable will be transformed to numeric variable and later we will examine 
# whether or not it belongs in model
albumtype.levels <- levels(dataset$`Album type`)
dataset$`Album type` <- as.numeric(dataset$`Album type`)
table(dataset$`Album type`)

# Because of the nature od 'Certification' variable, it will be deleted from dataset
dataset$Certification <- NULL

# plotting to see whether variables influence outcome variable
library(ggplot2)
ggplot(dataset, mapping = aes(x = `Year Recorded`, fill = OnChart)) + geom_density(alpha = 0.5)
# 'Year Recorded' will be excluded from model since the density function doesn't differ for songs that 
# have been on charts and for the songs that haven't been
ggplot(dataset, mapping = aes(x = `Year Released`, fill = OnChart)) + geom_density(alpha = 0.5)
# 'Year Released' will be excluded from model since the density function doesn't differ for songs that 
# have been on charts and for the songs that haven't been
ggplot(dataset, mapping = aes(x = `Album type`, fill = OnChart)) + geom_density(alpha = 0.5)
# 'Album type' affects whether a song has been on a chart
ggplot(dataset, mapping = aes(x = `Track number`, fill = OnChart)) + geom_density(alpha = 0.5)
# 'Track number' affects whether a song has been on a chart
ggplot(dataset, mapping = aes(x = `Song duration`, fill = OnChart)) + geom_density(alpha = 0.5)
# 'Song duration' affects whether a song has been on a chart
ggplot(dataset, mapping = aes(x = `acousticness`, fill = OnChart)) + geom_density(alpha = 0.5)
# 'acousticness' affects whether a song has been on a chart
ggplot(dataset, mapping = aes(x = `danceability`, fill = OnChart)) + geom_density(alpha = 0.5)
# 'danceability' affects whether a song has been on a chart
ggplot(dataset, mapping = aes(x = `energy`, fill = OnChart)) + geom_density(alpha = 0.5)
# 'energy' affects whether a song has been on a chart
ggplot(dataset, mapping = aes(x = `instrumentalness`, fill = OnChart)) + geom_density(alpha = 0.5)
# 'instrumentalness' affects whether a song has been on a chart
ggplot(dataset, mapping = aes(x = `liveness`, fill = OnChart)) + geom_density(alpha = 0.5)
# 'liveness' affects whether a song has been on a chart
ggplot(dataset, mapping = aes(x = `loudness`, fill = OnChart)) + geom_density(alpha = 0.5)
# 'loudness' affects whether a song has been on a chart
ggplot(dataset, mapping = aes(x = `speechiness`, fill = OnChart)) + geom_density(alpha = 0.5)
# 'speechiness' will be excluded from model since the density function doesn't differ for songs that 
# have been on charts and for the songs that haven't been
ggplot(dataset, mapping = aes(x = `tempo`, fill = OnChart)) + geom_density(alpha = 0.5)
# 'tempo' will be excluded from model since the density function doesn't differ for songs that 
# have been on charts and for the songs that haven't been
ggplot(dataset, mapping = aes(x = `valence`, fill = OnChart)) + geom_density(alpha = 0.5)
# 'valence' will be excluded from model since the density function doesn't differ for songs that 
# have been on charts and for the songs that haven't been

# excluding variables from model because of previously stated reasons
dataset$`Year Recorded` <- NULL
dataset$`Year Released` <- NULL
dataset$speechiness <- NULL
dataset$tempo <- NULL
dataset$valence <- NULL


# standardizing numeric variables
summary(dataset)
# checking if there are outliers
apply(dataset[,-ncol(dataset)], MARGIN = 2, FUN = function(x) length(boxplot.stats(x)$out))
# there are outliers so we will use standardization for rescaling the variables
# checking the variables' distribution with shapiro.test to see how to do it
apply(dataset[,-ncol(dataset)], MARGIN = 2, FUN = shapiro.test)
# only 'danceability' variable has Normal distribution

# standardizing not-normally distributed columns
dance.column <- which(colnames(dataset) == 'danceability')
dataset.stand <- dataset[,-c(dance.column, ncol(dataset))]
dataset.stand$`Track number` <- as.numeric(dataset.stand$`Track number`)
dataset.stand$`Song duration`<- as.numeric(dataset.stand$`Song duration`)

# apply the scalling function to each column
dataset.stand <- as.data.frame(apply(X = dataset.stand[,-1],
                     MARGIN = 2,
                     FUN = function(x) scale(x, center = median(x), scale = IQR(x))))

dataset.stand$`Album type` <- dataset$`Album type`

summary(dataset.stand)

# standardize 'danceability' variable
dataset.stand$danceability <- as.vector(scale(x = dataset$danceability, center = TRUE, scale = TRUE))

# adding outcome variable
dataset.stand$OnChart <- dataset$OnChart
# rearranging columns order
dataset.stand <- dataset.stand[,colnames(dataset)]

# creating train and test data sets
library(caret)
# set seed
set.seed(1)
# create train and test sets
train.indices <- createDataPartition(dataset.stand$OnChart, p = 0.8, list = FALSE)
train.data <- dataset.stand[train.indices,]
test.data <- dataset.stand[-train.indices,]

# determining best k with cross-validation 
library(e1071)
# define cross-validation (cv) parameters; we'll perform 10-fold cross-validation
numFolds <-  trainControl( method = "cv", number = 10)
# define the range for the k values to examine in the cross-validation
cpGrid <- expand.grid(.k = seq(from=3, to = 25, by = 2))
set.seed(1)
# run the cross-validation
knn.cv <- train(x = train.data[,-ncol(train.data)],
                y = train.data$OnChart,
                method = "knn",
                trControl = numFolds,
                tuneGrid = cpGrid)
knn.cv

k.best <- knn.cv$bestTune$k
# plot the cross-validation results
plot(knn.cv)
# best value for k is 9

# creating model with the best k
library(class)
knn.pred <- knn(train = train.data[,-ncol(train.data)],
                test = test.data[,-ncol(test.data)],
                cl = train.data$OnChart,
                k = k.best) 
head(knn.pred)

# create the confusion matrix
knn.cm <- table(true = test.data$OnChart, predicted = knn.pred)

knn.cm

# compute the evaluation metrics
knn.eval <- compute.eval.metrics(knn.cm)
knn.eval

# Accuracy is 81.35%, for 44 we have correctly predicted that the song won't be on any chart and for 4 songs we have
# correctly predicted that it will be on a chart

# Precision is 81.48%, for 44 out of 54 songs that we have predicted wouldn't be on any chart we were correct

# Recall is 97.7%, out of 45 songs that weren't on any charts for 44 we have predicted correctly that they wouldn't be

# F1 is 88.89%
