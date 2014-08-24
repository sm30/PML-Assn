---
title: "Practical Machine Learning-Data Science Specialization"
author: "Coursera Student"
date: "21 August 2014"
output: html_document
---
```{r setup, include=FALSE}
knitr::opts_chunk$set(cache=TRUE)
```
```{r reading data, echo = FALSE}
load("training.RData")
load("test.RData")
```
# PREPROCESSING DATA
```{r exlporation 0}
dim(training)
dim(test)
```
#### Are variable names unique?
```{r exlporation 1}
sum(table(names(training)))
sum(table(unique(names(training))))
sum(table(names(test)))
sum(table(unique(names(test))))
```
#### Do training and test variable names match?
```{r exlporation 2}
table(names(training)==names(test))
names(training)[names(training)!=names(test)]
names(test)[names(test)!=names(training)]
``` 
#### Are there missing values?
```{r exlporation 3}
getNa <- function(dfrm) lapply(dfrm, function(x) length(which(is.na(x) =="TRUE")))
g <- getNa(training)
# g[g!=0]
# length(g[g!=0])
g1 <- getNa(test)
# g1[g1!=0]
# length(g1[g1!=0])
```  
There are `r length(g[g!=0])` variables with missing values in the training set. Exactly 19216 rows are missing values from all of these variables.
There are `r length(g1[g1!=0])` variables with missing values in the test set. All 20 rows are missing values from all of these variables.

#### Do names of variables missing in training and test tally?
```{r exlporation 4, eval = FALSE}
table(names(g1[g1!=0]) %in% names(g[g!=0]))
``` 
Decided to remove all 100 variables from analyses. 67 variables are missing in all or
majority of cases in both training and test. If there was more data in these, imputation would be attempted. Remaining 33 variables that are missing in test cannot be included since predictions cannot be made on test with these.

#### Creating new training set by removing variables with missing values.
#### Removing first column containing row numbers and last with outcome, "classe".
#### Converting "user_name" and "cvtd_timestamp" to numeric factors
```{r exlporation 5}
n <- names(g1[g1!=0])
remnames <- names(training)[!(names(training) %in% n)]
newtrain <- training[,intersect(names(training),remnames)]
dim(newtrain)
newtrain <- newtrain[,-c(1,60)]
fac <- newtrain[,c(1,4)]
fac[,1] <- as.numeric(fac[,1])
fac[,2] <- as.numeric(fac[,2])
fac[,1] <- as.factor(fac[,1])
fac[,2] <- as.factor(fac[,2])
summary(fac)
newtrain <- newtrain[,-c(1,4)]
newtrain <- cbind(fac, newtrain)
dim(newtrain)
```  
The training and test data are from `r length(unique(newtrain[,2]))` people.

#### Filtering out variables with near zero variance
```{r exlporation 6, warning=FALSE}
library(caret)
nzv <- nearZeroVar(newtrain)
filtnewtrain <- newtrain[,-nzv]
dim(filtnewtrain)
```  
The variable `r setdiff(names(newtrain), names(filtnewtrain))` is removed by the near zero variance filter of caret package.

#### Converting data frame to numeric matrix 
#### Applying correlation filter - remove variables with correlation > 0.90
```{r exploration 7}
fil <- sapply(filtnewtrain, as.numeric)
corfil <- cor(fil)
f <- findCorrelation(corfil)
fil2 <- fil[,-f]
vars_to_remove <- setdiff(colnames(fil), colnames(fil2))
```
Variables "vars_to_remove" were removed due to high correlation.  
No linear combinations were found using appropriate function in caret package.

#### Subsetting same variables in test data and making new test set
```{r new test}
filtest <- test[,names(test)%in% colnames(fil2)]
dim(filtest)
filtest <- sapply(filtest, as.numeric)
```

#### Centering and scaling training and test data after removing "user_name" variable
#### Adding back "user_name" for both and "classe" for training data
#### Converting to data frame; "user_name", "classe" and "cvtd_timestamp" to factors
```{r preProcess}
preProcVal <- preProcess(fil2[,-1], method=c("center", "scale"))
trainF <- predict(preProcVal, fil2[,-1])
trainF <- cbind(fil2[,1], trainF)
colnames(trainF)[1:2] <- c("classe", "user_name")
trainF <- data.frame(trainF)
trainF[,1] <- as.factor(as.character(trainF[,1]))
trainF[,2] <- as.factor(as.character(trainF[,2]))
date <- strptime(training$cvtd_timestamp, "%d/%m/%Y %H:%M")
date <- as.factor(as.character(date))
trainF[,3] <- date
testF <- predict(preProcVal, filtest[,-1])
testF <- cbind(filtest[,1], testF)
colnames(testF)[1] <- "user_name"
testF <- data.frame(testF)
testF[,1] <- as.factor(as.character(testF[,1]))
date <- strptime(test$cvtd_timestamp, "%d/%m/%Y %H:%M")
date <- as.factor(as.character(date))
testF[,4] <- date
```
## EXPLORATORY ANALYSIS

 library(caret)

#### Timestamp
```{r timestamp}
featurePlot( x = trainF[,4:5],
             y = trainF$classe,
             plot = "density",
            auto.key = list(columns = 2))
```

#### Variables related to belt and arm
```{r box1}
featurePlot( x = trainF[,6:27],
             y = trainF$classe,
             plot = "box",
             auto.key = list(columns = 2))
```

#### Variables related to dumbell
```{r box2}
featurePlot( x = trainF[,c(28:31, 34:39)],
             y = trainF$classe,
             plot = "box",
            auto.key = list(columns = 2))
```                        

#### Variables related to forearm
```{r box 3}
featurePlot( x = trainF[,c(40:43,46:50)],
             y = trainF$classe,
             plot = "box",
             auto.key = list(columns = 2))
```

#### outliers in gyros_forearm_x and gyros_forearm_y, gyros_dumbell_y, gyros_dumbell_z          
```{r box 4}
featurePlot( x = trainF[,c(32,33,44,45)],
             y = trainF$classe,
             plot = "box",
             auto.key = list(columns = 2))
```

Single points are pulling up scale on y-axis to 100. Need to impute these values to median.

#### Pick out single extreme outliers and replace by median
```{r outlier}
summary(trainF[,c(32,33,44,45)])
w <- which(trainF$gyros_dumbbell_y > 80)
trainF$gyros_dumbbell_y[w] <- median(trainF$gyros_dumbbell_y)
w2 <- which(trainF$gyros_dumbbell_z > 100)
trainF$gyros_dumbbell_z[w2] <- median(trainF$gyros_dumbbell_z)
w3 <- which(abs(trainF$gyros_forearm_x) > 30)
trainF$gyros_forearm_x[w3] <- median(trainF$gyros_forearm_x)
w4 <- which(trainF$gyros_forearm_y > 99)
trainF$gyros_forearm_y[w4] <- median(trainF$gyros_forearm_y)
summary(trainF[,c(32,33,44,45)])
```

#### replacing numeric with original "classe"
```{r classe}
trainF$classe <- training$classe
```
## BUILDING THE MODEL

#### Splitting original training set into subtraining and internal test set
```{r split data}
library(caret)
set.seed(1234)
ind <- createDataPartition(trainF$classe, p=0.7, list=F)
subtrain <- trainF[ind,]
subtest <- trainF[-ind,]
```
Checked density distribution of time variables - structure not changed too much from original

#### Building the model : Random Forest with crossvalidation
```{r random forest, eval=FALSE}
library(doMC)
registerDoMC(cores = 5)
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
set.seed(1234)
rfFit <- train(classe~., data = subtrain, method = "rf",
trControl = fitControl, verbose= FALSE)
````
#### The model object
```
Random Forest 

13737 samples
   50 predictor
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Cross-Validated (10 fold, repeated 10 times) 

Summary of sample sizes: 12365, 12363, 12364, 12364, 12362, 12362, ... 

Resampling results across tuning parameters:

  mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
   2    0.985     0.981  0.003542     0.004484
  37    0.999     0.999  0.000771     0.000975
  72    0.999     0.998  0.000877     0.001109
```
Accuracy was used to select the optimal model using  the largest value.
The final value used for the model was mtry = 37. 

#### Predicting on held out test set belonging to original training data : 

#### In sample error
```{r insample predict, eval = FALSE}
predRF <- predict(rfFit, newdata = subtest)
confusionMatrix(predRF, subtest$classe)
```
```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1673    1    0    0    0
         B    1 1138    1    0    0
         C    0    0 1025    0    0
         D    0    0    0  964    0
         E    0    0    0    0 1082

Overall Statistics
                                          
               Accuracy : 0.9995          
                 95% CI : (0.9985, 0.9999)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9994          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9994   0.9991   0.9990   1.0000   1.0000
Specificity            0.9998   0.9996   1.0000   1.0000   1.0000
Pos Pred Value         0.9994   0.9982   1.0000   1.0000   1.0000
Neg Pred Value         0.9998   0.9998   0.9998   1.0000   1.0000
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2843   0.1934   0.1742   0.1638   0.1839
Detection Prevalence   0.2845   0.1937   0.1742   0.1638   0.1839
Balanced Accuracy      0.9996   0.9994   0.9995   1.0000   1.0000
```
The high accuracy values are indicative of overtraining. Random forest models themselves although powerful are prone to overfitting.  Out of sample prediction cannot be expected to be more accurate than 60-70%. 


