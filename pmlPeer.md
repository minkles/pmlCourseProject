Practical Machine Learning Course Project
========================================================

## Executive Summary

The goal of this project is to create a machine learning algorithm to predict, based on accelerometer data, whether or not an individual is doing an exercise properly or improperly. 

The training dataset contains accelerometer readings from 6 participants each performing one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in one of five ways:
A: Exactly according to the specification
B: Throwing the elbows to the front
C: Lifting the dumbbell only halfway
D: Lowering the dumbbell only halfway
E: Throwing the hips to the front.

My algorithm will predict, with over 98% accuracy, which of the five ways of lifting is being done, based on accelerometer readings from the arm, belt, forearm, and dumbbell itself.

###Data comes from the following source:

Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6. 

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3paO4q4fo

## Loading Required Packages and Data, and setting the seed

This algorithm makes use of functions from the caret, MASS, and kernlab packages. To be able to reproduce my results I set the seed to 1024.


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(MASS)
library(kernlab)
```

```
## Error: there is no package called 'kernlab'
```

```r
wleTrain <- read.csv("pml-training.csv")
wleTest <- read.csv("pml-testing.csv")
set.seed(1024)
```

## Subsetting

A summary of the data shows 120 variables in the training set. However, many of these variables are summary or index variables which will not be useful for building my algorithm. 

First, I remove the index variables.


```r
wleTrain <- wleTrain[,-(1:7)]
```

Next, I subset only the numerical variables, because the summary variables are factors and will not have an influence on the machine learning algorithm. The result variable, "classe", is also a factor, but I will add that back later. Before I do anything to wleTrain, I need to save the "classe" values to add back later.


```r
classe <- wleTrain$classe
wleTrain <- wleTrain[sapply(wleTrain, is.numeric)]
```

Next, I remove variables that contain NA values. These variables are calculated from the raw data, so they will not be necessary in the algorithm.


```r
wleTrain <- wleTrain[sapply(wleTrain, function(x) !any(is.na(x)))]
```

Finally, I add back the classe variable. When this is done, I have reduced the number of variables from 120 to 53 without losing any important data.


```r
wleTrain <- cbind(wleTrain,classe)
```

##Fitting the model

I chose to use a random forest model for its high accuracy rates and usefulness in predicting categorical data. 

I also chose to include Principal Components Analysis in my model because there is a large number of variables, and many of them are highly correlated with each other. This makes sense, because a single movement of a single sensor creates many raw variables: acceleration, gyroscope, and magnet reading in 3 directions, so we would expect these values to be highly correlated.


```r
M <- abs(cor(wleTrain[,-53]))
diag(M) <- 0
which(M > 0.9,arr.ind=T)
```

```
##                  row col
## total_accel_belt   4   1
## accel_belt_y       9   1
## accel_belt_z      10   1
## accel_belt_x       8   2
## roll_belt          1   4
## accel_belt_y       9   4
## accel_belt_z      10   4
## pitch_belt         2   8
## roll_belt          1   9
## total_accel_belt   4   9
## accel_belt_z      10   9
## roll_belt          1  10
## total_accel_belt   4  10
## accel_belt_y       9  10
## gyros_arm_y       19  18
## gyros_arm_x       18  19
## gyros_dumbbell_z  33  31
## gyros_forearm_z   46  31
## gyros_dumbbell_x  31  33
## gyros_forearm_z   46  33
## gyros_dumbbell_x  31  46
## gyros_dumbbell_z  33  46
```

My final model is as follows:


```r
modelFit <- train(wleTrain$classe ~ .,method="rf",preProcess="pca",data=wleTrain)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## Loading required namespace: e1071
```

```
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
```

##Cross validation

I chose to cross-validate using k-folds to ensure the model was not being overfitted. In my original cross-validation, I created 10 folds and found an average error rate of 1.7888%. However, due to the slow processing speed of random forests, for this writeup I will recreate the validation using only 5 folds. As you can see, the error rate is similar to the one I originally calculated. 

I expect my out-of-sample error rate to be close to 1.7888%.


```r
folds <- createFolds(y=wleTrain$classe,k=5,list=TRUE,returnTrain=TRUE)
kerrors <- vector()
for (i in 1:length(folds)) {
  inTrain <- folds[[i]]
  ktrain <- wleTrain[inTrain,]
  ktest <- wleTrain[-inTrain,]
  kFit <- train(ktrain$classe ~ ., method="rf",preProcess="pca",data=ktrain)
  predictions <- predict(kFit, newdata=ktest)
  kerror <- mean(predictions != ktest$classe)
  print (kerror)
  kerrors <- c(kerrors, kerror)
}
```

```
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
```

```
## [1] 0.01911
```

```
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
```

```
## [1] 0.01759
```

```
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
```

```
## [1] 0.01809
```

```
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
```

```
## [1] 0.02166
```

```
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
## Warning: invalid mtry: reset to within valid range
```

```
## [1] 0.02446
```

```r
errorRate <- mean(kerrors)
paste0("Average error rate: ", errorRate)
```

```
## [1] "Average error rate: 0.0201813830314467"
```
