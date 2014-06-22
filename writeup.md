Prediction Assignment Writeup
------------------------------

The first step was to look at the data and clean the data.

Removed user name, time based columns and other factor columns.
Removed rows that were marked as new window = yes
Removed mean, variance, std deviation, max, min, etc that were used in [original paper]: http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf for feature selection.

After observing the data it was very hard to determine pattern / correlation between predictors and classification due to number of predictors (52 after cleaning the data). 

So approach I took was to selected as small sample for initial experiment about 5% of test data and run train function without specifying method. 
Best method was identified using "random forest" algorithm. Accuracy was at about 80%.

After that I've selected larger data set of 20% for training set and left rest for testing:


```r
library(caret)

# Please set path to the folder you have downloaded CSV files to. Please note original CSV files were modified to clean up data.
setwd("D:/Install/R")

data <- read.csv("pml-training-no.csv")
set.seed(107)
inTraining <- createDataPartition(data$classe, p = 0.2, list = FALSE)
training <-data[inTraining,]
testing <-data[-inTraining,]
rfFit <- train(classe ~ ., data = training, method = "rf")
rfFit
```

```
## Random Forest 
## 
## 3846 samples
##   52 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 3846, 3846, 3846, 3846, 3846, 3846, ... 
## 
## Resampling results across tuning parameters:
## 
##  mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##  2     0.953     0.94   0.0055       0.00699 
##  27    0.959     0.948  0.0053       0.00672 
##  52    0.95      0.936  0.00901      0.0114  
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```


```r
rfFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 2.91%
## Confusion matrix:
##      A   B   C   D   E class.error
## A 1084   6   3   1   1     0.01005
## B   21 711   9   2   1     0.04435
## C    1  22 640   7   1     0.04620
## D    1   2  15 609   3     0.03333
## E    0   3   6   7 690     0.02266
```

Then I've used model to run cross validate model on the testing set:


```r
rfClasses <- predict(rfFit, newdata = testing)
```
  
Prediction on testing set (80% of test data) turned out to be quite good with Accuracy: 96% and (Random Forest specific out-of-bag) OOB estimate of error rate of 2.91%


```r
confusionMatrix(data = rfClasses, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4353  117    1    2    4
##          B   18 2799   95    9   35
##          C    0   49 2557   77    7
##          D    1    9   27 2425   27
##          E    4    0    1    4 2749
## 
## Overall Statistics
##                                         
##                Accuracy : 0.968         
##                  95% CI : (0.965, 0.971)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.96          
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.995    0.941    0.954    0.963    0.974
## Specificity             0.989    0.987    0.990    0.995    0.999
## Pos Pred Value          0.972    0.947    0.951    0.974    0.997
## Neg Pred Value          0.998    0.986    0.990    0.993    0.994
## Prevalence              0.285    0.193    0.174    0.164    0.184
## Detection Rate          0.283    0.182    0.166    0.158    0.179
## Detection Prevalence    0.291    0.192    0.175    0.162    0.179
## Balanced Accuracy       0.992    0.964    0.972    0.979    0.987
```

Next I've produced results for the validation data set:


```r
validateData <- read.csv("pml-testing.csv")
answers <- predict(rfFit, newdata = validateData)
answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

Submission of test cases confirmed 20/20 match on validation set that match well estimated model accuracy on the training/test set.

Although results were positive (achieved high accuracy) the downside of RandomForest is the fitted model is not human-readable.
It was hard to plot graphs or interpret outcome due to nature of the Forest tree algorithm. One of the ways to interpret model was to show importance of predictors to make some sense of the resulting model:


```r
importance(rfFit$finalModel)
```

```
##                      MeanDecreaseGini
## roll_belt                      375.48
## pitch_belt                     155.18
## yaw_belt                       197.94
## total_accel_belt                15.96
## gyros_belt_x                    15.46
## gyros_belt_y                    11.40
## gyros_belt_z                    38.55
## accel_belt_x                    13.95
## accel_belt_y                    13.81
## accel_belt_z                    59.14
## magnet_belt_x                   51.15
## magnet_belt_y                   71.51
## magnet_belt_z                   69.10
## roll_arm                        33.06
## pitch_arm                       19.94
## yaw_arm                         50.40
## total_accel_arm                 16.30
## gyros_arm_x                     18.80
## gyros_arm_y                     24.83
## gyros_arm_z                     11.08
## accel_arm_x                     28.93
## accel_arm_y                     23.32
## accel_arm_z                     15.24
## magnet_arm_x                    34.88
## magnet_arm_y                    32.67
## magnet_arm_z                    23.59
## roll_dumbbell                   87.32
## pitch_dumbbell                  23.67
## yaw_dumbbell                    45.36
## total_accel_dumbbell            62.33
## gyros_dumbbell_x                21.21
## gyros_dumbbell_y                36.48
## gyros_dumbbell_z                13.53
## accel_dumbbell_x                26.36
## accel_dumbbell_y                94.06
## accel_dumbbell_z                52.19
## magnet_dumbbell_x               69.71
## magnet_dumbbell_y              175.98
## magnet_dumbbell_z              168.24
## roll_forearm                   154.58
## pitch_forearm                  242.98
## yaw_forearm                     27.37
## total_accel_forearm             15.40
## gyros_forearm_x                 10.57
## gyros_forearm_y                 18.68
## gyros_forearm_z                 13.19
## accel_forearm_x                 83.71
## accel_forearm_y                 20.79
## accel_forearm_z                 40.73
## magnet_forearm_x                30.53
## magnet_forearm_y                32.93
## magnet_forearm_z                49.83
```

In order to improve on readability of the model an attempt was made to use rpart method of library(rpart) that produced single classification tree potentially readable. The results were not promising with cross validation accuracy of 30% that was fairly low compare to Random forest result.

Another attempt was made to use "party" library and ctree algorithm that as well output single classification tree:


```r
inTraining <- createDataPartition(data$classe, p = 0.5, list = FALSE)
training <-data[inTraining,]
testing <-data[-inTraining,]
ctFit <- ctree(classe ~ ., data = training)
ctFit
```

```
## 
## 	 Conditional inference tree with 317 terminal nodes
## 
## Response:  classe 
## Inputs:  roll_belt, pitch_belt, yaw_belt, total_accel_belt, gyros_belt_x, gyros_belt_y, gyros_belt_z, accel_belt_x, accel_belt_y, accel_belt_z, magnet_belt_x, magnet_belt_y, magnet_belt_z, roll_arm, pitch_arm, yaw_arm, total_accel_arm, gyros_arm_x, gyros_arm_y, gyros_arm_z, accel_arm_x, accel_arm_y, accel_arm_z, magnet_arm_x, magnet_arm_y, magnet_arm_z, roll_dumbbell, pitch_dumbbell, yaw_dumbbell, total_accel_dumbbell, gyros_dumbbell_x, gyros_dumbbell_y, gyros_dumbbell_z, accel_dumbbell_x, accel_dumbbell_y, accel_dumbbell_z, magnet_dumbbell_x, magnet_dumbbell_y, magnet_dumbbell_z, roll_forearm, pitch_forearm, yaw_forearm, total_accel_forearm, gyros_forearm_x, gyros_forearm_y, gyros_forearm_z, accel_forearm_x, accel_forearm_y, accel_forearm_z, magnet_forearm_x, magnet_forearm_y, magnet_forearm_z 
## Number of observations:  9609 
## 
## 1) pitch_forearm <= -33.7; criterion = 1, statistic = 1883.33
##   2) gyros_dumbbell_y <= 0.51; criterion = 1, statistic = 64.094
##     3) gyros_arm_y <= -0.82; criterion = 1, statistic = 53.774
##       4)*  weights = 22 
##     3) gyros_arm_y > -0.82
##       5)*  weights = 790 
##   2) gyros_dumbbell_y > 0.51
##     6)*  weights = 10 
## 1) pitch_forearm > -33.7
##   7) magnet_belt_y <= 557; criterion = 1, statistic = 1004.72
##     8) magnet_dumbbell_z <= 146; criterion = 1, statistic = 431.513
##       9) magnet_forearm_z <= -332; criterion = 1, statistic = 96.196
##         10)*  weights = 9 
##       9) magnet_forearm_z > -332
##         11) gyros_dumbbell_x <= 1.14; criterion = 0.997, statistic = 19.633
##           12)*  weights = 530 
##         11) gyros_dumbbell_x > 1.14
##           13)*  weights = 7 
##     8) magnet_dumbbell_z > 146
##       14) magnet_belt_z <= -434; criterion = 1, statistic = 112.643
##         15) yaw_belt <= -87.5; criterion = 1, statistic = 57.344
##           16)*  weights = 13 
##         15) yaw_belt > -87.5
##           17)*  weights = 118 
##       14) magnet_belt_z > -434
##         18)*  weights = 34 
##   7) magnet_belt_y > 557
##     19) roll_dumbbell <= -95.5; criterion = 1, statistic = 831.254
##       20) magnet_belt_x <= 46; criterion = 1, statistic = 194.92
##         21) gyros_belt_z <= 0.11; criterion = 1, statistic = 203.944
##           22) total_accel_forearm <= 6; criterion = 1, statistic = 140.876
##             23) gyros_arm_x <= 0.22; criterion = 1, statistic = 32.636
##               24)*  weights = 9 
##             23) gyros_arm_x > 0.22
##               25)*  weights = 26 
##           22) total_accel_forearm > 6
##             26) yaw_belt <= 4.96; criterion = 1, statistic = 142.825
##               27) magnet_dumbbell_y <= 91; criterion = 1, statistic = 127.463
##                 28) pitch_arm <= 16.3; criterion = 0.983, statistic = 12.944
##                   29)*  weights = 34 
##                 28) pitch_arm > 16.3
##                   30)*  weights = 18 
##               27) magnet_dumbbell_y > 91
##                 31) total_accel_dumbbell <= 4; criterion = 1, statistic = 157.515
##                   32) magnet_dumbbell_x <= -564; criterion = 1, statistic = 148.481
##                     33) magnet_dumbbell_z <= -11; criterion = 1, statistic = 83.846
##                       34)*  weights = 24 
##                     33) magnet_dumbbell_z > -11
##                       35) yaw_belt <= -93.5; criterion = 1, statistic = 31.401
##                         36)*  weights = 10 
##                       35) yaw_belt > -93.5
##                         37)*  weights = 118 
##                   32) magnet_dumbbell_x > -564
##                     38) yaw_belt <= 1.03; criterion = 1, statistic = 48.22
##                       39) magnet_arm_z <= -81; criterion = 1, statistic = 33.339
##                         40) roll_forearm <= -9.83; criterion = 0.961, statistic = 14.362
##                           41)*  weights = 27 
##                         40) roll_forearm > -9.83
##                           42)*  weights = 9 
##                       39) magnet_arm_z > -81
##                         43) accel_forearm_z <= 141; criterion = 0.992, statistic = 20.33
##                           44) pitch_arm <= -23.8; criterion = 0.999, statistic = 22.796
##                             45)*  weights = 16 
##                           44) pitch_arm > -23.8
##                             46)*  weights = 17 
##                         43) accel_forearm_z > 141
##                           47) magnet_dumbbell_y <= 223; criterion = 1, statistic = 32.834
##                             48)*  weights = 9 
##                           47) magnet_dumbbell_y > 223
##                             49) gyros_dumbbell_x <= 0.11; criterion = 1, statistic = 24.851
##                               50)*  weights = 8 
##                             49) gyros_dumbbell_x > 0.11
##                               51)*  weights = 31 
##                     38) yaw_belt > 1.03
##                       52)*  weights = 15 
##                 31) total_accel_dumbbell > 4
##                   53) pitch_dumbbell <= -1.517; criterion = 1, statistic = 70.055
##                     54) accel_dumbbell_z <= -3; criterion = 1, statistic = 42.929
##                       55) magnet_dumbbell_z <= -37; criterion = 1, statistic = 72.187
##                         56)*  weights = 18 
##                       55) magnet_dumbbell_z > -37
##                         57) magnet_dumbbell_y <= 152; criterion = 1, statistic = 47.791
##                           58)*  weights = 75 
##                         57) magnet_dumbbell_y > 152
##                           59)*  weights = 15 
##                     54) accel_dumbbell_z > -3
##                       60) magnet_dumbbell_z <= -34; criterion = 1, statistic = 43.399
##                         61)*  weights = 8 
##                       60) magnet_dumbbell_z > -34
##                         62) gyros_belt_x <= -0.05; criterion = 1, statistic = 29.461
##                           63)*  weights = 10 
##                         62) gyros_belt_x > -0.05
##                           64)*  weights = 269 
##                   53) pitch_dumbbell > -1.517
##                     65)*  weights = 18 
##             26) yaw_belt > 4.96
##               66)*  weights = 28 
##         21) gyros_belt_z > 0.11
##           67)*  weights = 54 
##       20) magnet_belt_x > 46
##         68) magnet_dumbbell_y <= 297; criterion = 1, statistic = 62.622
##           69) magnet_arm_z <= 502; criterion = 1, statistic = 46.112
##             70)*  weights = 7 
##           69) magnet_arm_z > 502
##             71)*  weights = 44 
##         68) magnet_dumbbell_y > 297
##           72)*  weights = 21 
##     19) roll_dumbbell > -95.5
##       73) magnet_dumbbell_z <= 48; criterion = 1, statistic = 731.106
##         74) pitch_forearm <= 35; criterion = 1, statistic = 527.767
##           75) magnet_arm_z <= -200; criterion = 1, statistic = 382.725
##             76) total_accel_forearm <= 46; criterion = 1, statistic = 111.217
##               77) magnet_dumbbell_z <= -8; criterion = 1, statistic = 74.605
##                 78) magnet_forearm_z <= 272; criterion = 1, statistic = 46.584
##                   79) roll_belt <= 125; criterion = 1, statistic = 22.933
##                     80)*  weights = 8 
##                   79) roll_belt > 125
##                     81)*  weights = 16 
##                 78) magnet_forearm_z > 272
##                   82) roll_forearm <= 137; criterion = 1, statistic = 23.109
##                     83)*  weights = 94 
##                   82) roll_forearm > 137
##                     84)*  weights = 11 
##               77) magnet_dumbbell_z > -8
##                 85) roll_dumbbell <= -3.948; criterion = 1, statistic = 35.999
##                   86)*  weights = 16 
##                 85) roll_dumbbell > -3.948
##                   87) total_accel_forearm <= 31; criterion = 0.999, statistic = 27.254
##                     88)*  weights = 12 
##                   87) total_accel_forearm > 31
##                     89) magnet_arm_x <= 457; criterion = 0.956, statistic = 16.571
##                       90)*  weights = 7 
##                     89) magnet_arm_x > 457
##                       91)*  weights = 31 
##             76) total_accel_forearm > 46
##               92) roll_belt <= 126; criterion = 1, statistic = 43.136
##                 93) accel_arm_x <= 148; criterion = 0.999, statistic = 21.856
##                   94)*  weights = 7 
##                 93) accel_arm_x > 148
##                   95) yaw_belt <= -0.02; criterion = 1, statistic = 20.993
##                     96)*  weights = 15 
##                   95) yaw_belt > -0.02
##                     97)*  weights = 7 
##               92) roll_belt > 126
##                 98)*  weights = 23 
##           75) magnet_arm_z > -200
##             99) magnet_dumbbell_y <= 457; criterion = 1, statistic = 357.429
##               100) roll_forearm <= 115; criterion = 1, statistic = 317.268
##                 101) roll_forearm <= -54.4; criterion = 1, statistic = 287.476
##                   102) yaw_belt <= -93.2; criterion = 1, statistic = 67.591
##                     103)*  weights = 31 
##                   102) yaw_belt > -93.2
##                     104) yaw_belt <= -0.06; criterion = 1, statistic = 33.355
##                       105) yaw_forearm <= -73.8; criterion = 0.999, statistic = 25.566
##                         106)*  weights = 12 
##                       105) yaw_forearm > -73.8
##                         107)*  weights = 15 
##                     104) yaw_belt > -0.06
##                       108)*  weights = 39 
##                 101) roll_forearm > -54.4
##                   109) magnet_dumbbell_z <= 33; criterion = 1, statistic = 265.099
##                     110) yaw_dumbbell <= -114.9; criterion = 1, statistic = 145.011
##                       111) accel_forearm_x <= -377; criterion = 0.968, statistic = 19.502
##                         112)*  weights = 8 
##                       111) accel_forearm_x > -377
##                         113)*  weights = 13 
##                     110) yaw_dumbbell > -114.9
##                       114) yaw_dumbbell <= -99.71; criterion = 1, statistic = 123.106
##                         115) total_accel_arm <= 9; criterion = 1, statistic = 85.767
##                           116)*  weights = 14 
##                         115) total_accel_arm > 9
##                           117) gyros_arm_y <= -1.53; criterion = 1, statistic = 76.667
##                             118)*  weights = 8 
##                           117) gyros_arm_y > -1.53
##                             119) gyros_dumbbell_y <= -0.08; criterion = 1, statistic = 61.289
##                               120) yaw_belt <= 165; criterion = 0.996, statistic = 18.856
##                                 121)*  weights = 13 
##                               120) yaw_belt > 165
##                                 122)*  weights = 7 
##                             119) gyros_dumbbell_y > -0.08
##                               123) gyros_dumbbell_z <= 0.1; criterion = 1, statistic = 53.392
##                                 124) gyros_arm_y <= -0.02; criterion = 0.999, statistic = 25.515
##                                   125) accel_forearm_x <= -189; criterion = 1, statistic = 26.474
##                                     126)*  weights = 13 
##                                   125) accel_forearm_x > -189
##                                     127)*  weights = 132 
##                                 124) gyros_arm_y > -0.02
##                                   128)*  weights = 9 
##                               123) gyros_dumbbell_z > 0.1
##                                 129)*  weights = 8 
##                       114) yaw_dumbbell > -99.71
##                         130) magnet_dumbbell_z <= 27; criterion = 1, statistic = 92.425
##                           131) roll_dumbbell <= -90.96; criterion = 1, statistic = 77.995
##                             132)*  weights = 19 
##                           131) roll_dumbbell > -90.96
##                             133) magnet_arm_x <= -159; criterion = 1, statistic = 60.103
##                               134) gyros_dumbbell_y <= 0.64; criterion = 1, statistic = 71.915
##                                 135) magnet_forearm_z <= 233; criterion = 0.995, statistic = 18.568
##                                   136)*  weights = 11 
##                                 135) magnet_forearm_z > 233
##                                   137) gyros_arm_x <= 1.41; criterion = 0.991, statistic = 17.387
##                                     138)*  weights = 395 
##                                   137) gyros_arm_x > 1.41
##                                     139)*  weights = 9 
##                               134) gyros_dumbbell_y > 0.64
##                                 140)*  weights = 8 
##                             133) magnet_arm_x > -159
##                               141) gyros_belt_y <= 0.08; criterion = 1, statistic = 41.314
##                                 142) accel_forearm_z <= -10; criterion = 0.998, statistic = 20.422
##                                   143) roll_forearm <= 91.2; criterion = 0.971, statistic = 11.903
##                                     144)*  weights = 149 
##                                   143) roll_forearm > 91.2
##                                     145) total_accel_dumbbell <= 2; criterion = 1, statistic = 26.72
##                                       146)*  weights = 7 
##                                     145) total_accel_dumbbell > 2
##                                       147) magnet_belt_x <= -2; criterion = 0.993, statistic = 14.564
##                                         148)*  weights = 9 
##                                       147) magnet_belt_x > -2
##                                         149)*  weights = 44 
##                                 142) accel_forearm_z > -10
##                                   150)*  weights = 7 
##                               141) gyros_belt_y > 0.08
##                                 151) magnet_dumbbell_x <= -438; criterion = 1, statistic = 129.801
##                                   152) magnet_dumbbell_y <= 379; criterion = 1, statistic = 83.635
##                                     153) magnet_forearm_z <= 671; criterion = 1, statistic = 26.74
##                                       154) gyros_arm_y <= -1.35; criterion = 1, statistic = 29.815
##                                         155)*  weights = 11 
##                                       154) gyros_arm_y > -1.35
##                                         156) roll_belt <= 122; criterion = 0.999, statistic = 22.288
##                                           157) roll_forearm <= 27.8; criterion = 0.979, statistic = 12.46
##                                             158)*  weights = 219 
##                                           157) roll_forearm > 27.8
##                                             159) yaw_belt <= 0.31; criterion = 1, statistic = 22.181
##                                               160)*  weights = 7 
##                                             159) yaw_belt > 0.31
##                                               161)*  weights = 19 
##                                         156) roll_belt > 122
##                                           162)*  weights = 12 
##                                     153) magnet_forearm_z > 671
##                                       163) accel_forearm_x <= -162; criterion = 0.997, statistic = 19.865
##                                         164)*  weights = 46 
##                                       163) accel_forearm_x > -162
##                                         165)*  weights = 18 
##                                   152) magnet_dumbbell_y > 379
##                                     166) roll_dumbbell <= -28.07; criterion = 1, statistic = 44.03
##                                       167)*  weights = 7 
##                                     166) roll_dumbbell > -28.07
##                                       168) magnet_dumbbell_z <= -188; criterion = 1, statistic = 26.26
##                                         169)*  weights = 15 
##                                       168) magnet_dumbbell_z > -188
##                                         170) yaw_dumbbell <= 83.6; criterion = 0.997, statistic = 19.769
##                                           171)*  weights = 21 
##                                         170) yaw_dumbbell > 83.6
##                                           172)*  weights = 39 
##                                 151) magnet_dumbbell_x > -438
##                                   173) accel_dumbbell_z <= 2; criterion = 0.998, statistic = 20.279
##                                     174)*  weights = 15 
##                                   173) accel_dumbbell_z > 2
##                                     175)*  weights = 7 
##                         130) magnet_dumbbell_z > 27
##                           176) pitch_arm <= 68.3; criterion = 1, statistic = 28.493
##                             177) roll_arm <= -14.2; criterion = 1, statistic = 23.114
##                               178)*  weights = 15 
##                             177) roll_arm > -14.2
##                               179) gyros_dumbbell_y <= 0.05; criterion = 0.985, statistic = 16.274
##                                 180)*  weights = 15 
##                               179) gyros_dumbbell_y > 0.05
##                                 181)*  weights = 15 
##                           176) pitch_arm > 68.3
##                             182)*  weights = 10 
##                   109) magnet_dumbbell_z > 33
##                     183) pitch_arm <= 42.3; criterion = 1, statistic = 116.933
##                       184) roll_arm <= 12.8; criterion = 1, statistic = 73.578
##                         185) accel_arm_x <= -110; criterion = 1, statistic = 72.941
##                           186) roll_arm <= -17; criterion = 1, statistic = 29.454
##                             187)*  weights = 9 
##                           186) roll_arm > -17
##                             188)*  weights = 59 
##                         185) accel_arm_x > -110
##                           189) magnet_dumbbell_y <= 264; criterion = 1, statistic = 42.264
##                             190) gyros_arm_x <= -0.66; criterion = 0.987, statistic = 13.394
##                               191) gyros_dumbbell_y <= 0; criterion = 0.996, statistic = 15.657
##                                 192)*  weights = 12 
##                               191) gyros_dumbbell_y > 0
##                                 193)*  weights = 8 
##                             190) gyros_arm_x > -0.66
##                               194)*  weights = 8 
##                           189) magnet_dumbbell_y > 264
##                             195) roll_arm <= -4.98; criterion = 1, statistic = 28.805
##                               196) yaw_belt <= 139; criterion = 1, statistic = 47.651
##                                 197)*  weights = 10 
##                               196) yaw_belt > 139
##                                 198) yaw_belt <= 165; criterion = 1, statistic = 29.875
##                                   199)*  weights = 14 
##                                 198) yaw_belt > 165
##                                   200) pitch_belt <= -42.1; criterion = 0.999, statistic = 24
##                                     201)*  weights = 12 
##                                   200) pitch_belt > -42.1
##                                     202)*  weights = 13 
##                             195) roll_arm > -4.98
##                               203)*  weights = 13 
##                       184) roll_arm > 12.8
##                         204) magnet_forearm_y <= 664; criterion = 0.999, statistic = 22.221
##                           205)*  weights = 7 
##                         204) magnet_forearm_y > 664
##                           206) gyros_arm_x <= 0.45; criterion = 0.998, statistic = 17.322
##                             207)*  weights = 7 
##                           206) gyros_arm_x > 0.45
##                             208)*  weights = 15 
##                     183) pitch_arm > 42.3
##                       209) yaw_belt <= 171; criterion = 1, statistic = 28.033
##                         210) magnet_forearm_y <= 780; criterion = 1, statistic = 26.458
##                           211) magnet_forearm_y <= 729; criterion = 0.999, statistic = 18.816
##                             212)*  weights = 7 
##                           211) magnet_forearm_y > 729
##                             213)*  weights = 13 
##                         210) magnet_forearm_y > 780
##                           214)*  weights = 8 
##                       209) yaw_belt > 171
##                         215)*  weights = 45 
##               100) roll_forearm > 115
##                 216) roll_forearm <= 138; criterion = 1, statistic = 262.273
##                   217) pitch_dumbbell <= 9.632; criterion = 1, statistic = 90.085
##                     218) magnet_forearm_z <= 438; criterion = 1, statistic = 52.317
##                       219)*  weights = 7 
##                     218) magnet_forearm_z > 438
##                       220) magnet_forearm_z <= 568; criterion = 1, statistic = 40.928
##                         221) gyros_arm_x <= -2.25; criterion = 0.985, statistic = 18.92
##                           222)*  weights = 9 
##                         221) gyros_arm_x > -2.25
##                           223)*  weights = 14 
##                       220) magnet_forearm_z > 568
##                         224) roll_forearm <= 123; criterion = 0.996, statistic = 21.842
##                           225) total_accel_dumbbell <= 29; criterion = 1, statistic = 21.079
##                             226)*  weights = 20 
##                           225) total_accel_dumbbell > 29
##                             227)*  weights = 11 
##                         224) roll_forearm > 123
##                           228)*  weights = 68 
##                   217) pitch_dumbbell > 9.632
##                     229) magnet_belt_x <= -3; criterion = 1, statistic = 67.085
##                       230) roll_forearm <= 129; criterion = 1, statistic = 23.899
##                         231)*  weights = 19 
##                       230) roll_forearm > 129
##                         232)*  weights = 10 
##                     229) magnet_belt_x > -3
##                       233) gyros_arm_y <= -1.38; criterion = 1, statistic = 45.522
##                         234)*  weights = 7 
##                       233) gyros_arm_y > -1.38
##                         235) roll_arm <= 46.3; criterion = 1, statistic = 48.076
##                           236)*  weights = 7 
##                         235) roll_arm > 46.3
##                           237) accel_dumbbell_y <= 19; criterion = 1, statistic = 50.088
##                             238) magnet_belt_x <= 10; criterion = 0.999, statistic = 19.139
##                               239) gyros_arm_y <= 0.48; criterion = 0.982, statistic = 12.831
##                                 240)*  weights = 44 
##                               239) gyros_arm_y > 0.48
##                                 241) roll_arm <= 65; criterion = 0.964, statistic = 11.456
##                                   242)*  weights = 9 
##                                 241) roll_arm > 65
##                                   243)*  weights = 11 
##                             238) magnet_belt_x > 10
##                               244) total_accel_forearm <= 35; criterion = 0.959, statistic = 11.218
##                                 245)*  weights = 7 
##                               244) total_accel_forearm > 35
##                                 246)*  weights = 19 
##                           237) accel_dumbbell_y > 19
##                             247)*  weights = 8 
##                 216) roll_forearm > 138
##                   248) yaw_dumbbell <= 56.4; criterion = 1, statistic = 102.111
##                     249) yaw_arm <= 0; criterion = 1, statistic = 26.842
##                       250)*  weights = 25 
##                     249) yaw_arm > 0
##                       251) accel_arm_y <= 50; criterion = 0.993, statistic = 17.731
##                         252)*  weights = 9 
##                       251) accel_arm_y > 50
##                         253)*  weights = 22 
##                   248) yaw_dumbbell > 56.4
##                     254) magnet_arm_z <= 149; criterion = 1, statistic = 52.956
##                       255)*  weights = 22 
##                     254) magnet_arm_z > 149
##                       256) gyros_arm_z <= -0.36; criterion = 1, statistic = 45.805
##                         257)*  weights = 24 
##                       256) gyros_arm_z > -0.36
##                         258) accel_arm_y <= 142; criterion = 1, statistic = 21.114
##                           259) magnet_belt_y <= 578; criterion = 1, statistic = 27.719
##                             260)*  weights = 8 
##                           259) magnet_belt_y > 578
##                             261)*  weights = 152 
##                         258) accel_arm_y > 142
##                           262)*  weights = 8 
##             99) magnet_dumbbell_y > 457
##               263) gyros_belt_z <= -0.41; criterion = 1, statistic = 145.207
##                 264)*  weights = 9 
##               263) gyros_belt_z > -0.41
##                 265) magnet_belt_z <= -327; criterion = 1, statistic = 153.818
##                   266) magnet_dumbbell_y <= 501; criterion = 1, statistic = 35.75
##                     267) accel_dumbbell_z <= 2; criterion = 0.999, statistic = 19.573
##                       268)*  weights = 28 
##                     267) accel_dumbbell_z > 2
##                       269)*  weights = 11 
##                   266) magnet_dumbbell_y > 501
##                     270)*  weights = 9 
##                 265) magnet_belt_z > -327
##                   271) magnet_belt_z <= -291; criterion = 1, statistic = 91.566
##                     272) accel_dumbbell_z <= -198; criterion = 1, statistic = 43.903
##                       273)*  weights = 7 
##                     272) accel_dumbbell_z > -198
##                       274) total_accel_dumbbell <= 5; criterion = 1, statistic = 24.98
##                         275) yaw_belt <= -2.89; criterion = 1, statistic = 25.817
##                           276)*  weights = 33 
##                         275) yaw_belt > -2.89
##                           277)*  weights = 11 
##                       274) total_accel_dumbbell > 5
##                         278)*  weights = 180 
##                   271) magnet_belt_z > -291
##                     279)*  weights = 7 
##         74) pitch_forearm > 35
##           280) total_accel_dumbbell <= 21; criterion = 1, statistic = 265.635
##             281) total_accel_dumbbell <= 11; criterion = 1, statistic = 244.358
##               282) magnet_belt_x <= 32; criterion = 1, statistic = 236.351
##                 283) accel_forearm_x <= -170; criterion = 1, statistic = 234.976
##                   284) magnet_dumbbell_z <= 3; criterion = 1, statistic = 113.764
##                     285) magnet_dumbbell_z <= -170; criterion = 1, statistic = 100.989
##                       286) gyros_dumbbell_x <= -0.14; criterion = 0.987, statistic = 16.517
##                         287)*  weights = 13 
##                       286) gyros_dumbbell_x > -0.14
##                         288)*  weights = 7 
##                     285) magnet_dumbbell_z > -170
##                       289) yaw_dumbbell <= 44.38; criterion = 1, statistic = 92.631
##                         290) accel_dumbbell_y <= 56; criterion = 1, statistic = 77.684
##                           291) accel_dumbbell_x <= -21; criterion = 1, statistic = 59.093
##                             292)*  weights = 8 
##                           291) accel_dumbbell_x > -21
##                             293) pitch_belt <= 11.6; criterion = 0.998, statistic = 20.586
##                               294) total_accel_belt <= 4; criterion = 1, statistic = 40.485
##                                 295) gyros_belt_x <= 0.02; criterion = 0.999, statistic = 19.399
##                                   296)*  weights = 27 
##                                 295) gyros_belt_x > 0.02
##                                   297)*  weights = 14 
##                               294) total_accel_belt > 4
##                                 298)*  weights = 7 
##                             293) pitch_belt > 11.6
##                               299)*  weights = 196 
##                         290) accel_dumbbell_y > 56
##                           300)*  weights = 7 
##                       289) yaw_dumbbell > 44.38
##                         301) accel_forearm_z <= -126; criterion = 1, statistic = 154.014
##                           302) magnet_arm_x <= -397; criterion = 1, statistic = 74.794
##                             303)*  weights = 7 
##                           302) magnet_arm_x > -397
##                             304) accel_forearm_y <= 56; criterion = 1, statistic = 52.748
##                               305) yaw_forearm <= -117; criterion = 1, statistic = 30.945
##                                 306) pitch_forearm <= 56.8; criterion = 0.993, statistic = 22.979
##                                   307)*  weights = 36 
##                                 306) pitch_forearm > 56.8
##                                   308)*  weights = 13 
##                               305) yaw_forearm > -117
##                                 309)*  weights = 32 
##                             304) accel_forearm_y > 56
##                               310) pitch_forearm <= 62.4; criterion = 1, statistic = 58.563
##                                 311) total_accel_arm <= 24; criterion = 1, statistic = 35.764
##                                   312) gyros_dumbbell_x <= 0.21; criterion = 1, statistic = 31.674
##                                     313) magnet_dumbbell_y <= 301; criterion = 0.998, statistic = 22.724
##                                       314)*  weights = 7 
##                                     313) magnet_dumbbell_y > 301
##                                       315) magnet_dumbbell_z <= -28; criterion = 0.961, statistic = 11.337
##                                         316)*  weights = 7 
##                                       315) magnet_dumbbell_z > -28
##                                         317)*  weights = 18 
##                                   312) gyros_dumbbell_x > 0.21
##                                     318)*  weights = 11 
##                                 311) total_accel_arm > 24
##                                   319) magnet_belt_y <= 602; criterion = 0.979, statistic = 18.201
##                                     320) gyros_belt_z <= -0.38; criterion = 1, statistic = 25.317
##                                       321)*  weights = 84 
##                                     320) gyros_belt_z > -0.38
##                                       322)*  weights = 10 
##                                   319) magnet_belt_y > 602
##                                     323)*  weights = 7 
##                               310) pitch_forearm > 62.4
##                                 324)*  weights = 13 
##                         301) accel_forearm_z > -126
##                           325) roll_belt <= 116; criterion = 1, statistic = 20.408
##                             326)*  weights = 19 
##                           325) roll_belt > 116
##                             327)*  weights = 9 
##                   284) magnet_dumbbell_z > 3
##                     328)*  weights = 8 
##                 283) accel_forearm_x > -170
##                   329) magnet_arm_x <= -265; criterion = 1, statistic = 111.046
##                     330) roll_arm <= -144; criterion = 1, statistic = 54.419
##                       331)*  weights = 8 
##                     330) roll_arm > -144
##                       332) magnet_arm_y <= 345; criterion = 1, statistic = 43.643
##                         333) magnet_dumbbell_z <= -9; criterion = 0.982, statistic = 18.477
##                           334)*  weights = 17 
##                         333) magnet_dumbbell_z > -9
##                           335)*  weights = 11 
##                       332) magnet_arm_y > 345
##                         336) magnet_belt_x <= 1; criterion = 1, statistic = 28.711
##                           337) magnet_dumbbell_x <= 434; criterion = 0.952, statistic = 13.935
##                             338)*  weights = 7 
##                           337) magnet_dumbbell_x > 434
##                             339)*  weights = 23 
##                         336) magnet_belt_x > 1
##                           340) yaw_dumbbell <= 85.98; criterion = 1, statistic = 19.788
##                             341)*  weights = 7 
##                           340) yaw_dumbbell > 85.98
##                             342) yaw_belt <= -6.02; criterion = 0.999, statistic = 17.596
##                               343)*  weights = 17 
##                             342) yaw_belt > -6.02
##                               344)*  weights = 7 
##                   329) magnet_arm_x > -265
##                     345) pitch_forearm <= 59.3; criterion = 1, statistic = 88.355
##                       346) accel_forearm_x <= -78; criterion = 1, statistic = 86.262
##                         347) magnet_arm_x <= 183; criterion = 1, statistic = 53.013
##                           348) pitch_arm <= -15; criterion = 0.999, statistic = 28.431
##                             349) roll_arm <= 117; criterion = 0.995, statistic = 15.348
##                               350)*  weights = 7 
##                             349) roll_arm > 117
##                               351)*  weights = 14 
##                           348) pitch_arm > -15
##                             352) gyros_dumbbell_y <= -0.14; criterion = 0.985, statistic = 21.218
##                               353)*  weights = 7 
##                             352) gyros_dumbbell_y > -0.14
##                               354)*  weights = 21 
##                         347) magnet_arm_x > 183
##                           355) accel_forearm_y <= -57; criterion = 1, statistic = 32.376
##                             356) roll_forearm <= -134; criterion = 1, statistic = 48.507
##                               357) gyros_arm_x <= 1; criterion = 1, statistic = 39.116
##                                 358)*  weights = 15 
##                               357) gyros_arm_x > 1
##                                 359)*  weights = 30 
##                             356) roll_forearm > -134
##                               360) roll_arm <= 53.2; criterion = 0.997, statistic = 16.141
##                                 361)*  weights = 7 
##                               360) roll_arm > 53.2
##                                 362)*  weights = 15 
##                           355) accel_forearm_y > -57
##                             363) accel_forearm_x <= -102; criterion = 0.994, statistic = 23.161
##                               364) gyros_arm_y <= 1.38; criterion = 0.991, statistic = 22.198
##                                 365) gyros_arm_y <= -1.25; criterion = 0.993, statistic = 22.827
##                                   366)*  weights = 11 
##                                 365) gyros_arm_y > -1.25
##                                   367) gyros_belt_z <= 0; criterion = 0.978, statistic = 20.371
##                                     368) gyros_dumbbell_y <= -0.03; criterion = 0.994, statistic = 20.787
##                                       369)*  weights = 55 
##                                     368) gyros_dumbbell_y > -0.03
##                                       370) gyros_dumbbell_y <= 0; criterion = 0.955, statistic = 16.546
##                                         371)*  weights = 11 
##                                       370) gyros_dumbbell_y > 0
##                                         372)*  weights = 30 
##                                   367) gyros_belt_z > 0
##                                     373)*  weights = 7 
##                               364) gyros_arm_y > 1.38
##                                 374)*  weights = 10 
##                             363) accel_forearm_x > -102
##                               375)*  weights = 11 
##                       346) accel_forearm_x > -78
##                         376) magnet_belt_x <= -1; criterion = 1, statistic = 51.133
##                           377) gyros_arm_x <= -3.31; criterion = 0.996, statistic = 23.961
##                             378) roll_arm <= 107; criterion = 0.999, statistic = 19.246
##                               379)*  weights = 7 
##                             378) roll_arm > 107
##                               380)*  weights = 21 
##                           377) gyros_arm_x > -3.31
##                             381)*  weights = 18 
##                         376) magnet_belt_x > -1
##                           382) yaw_dumbbell <= -73.89; criterion = 1, statistic = 28.38
##                             383)*  weights = 15 
##                           382) yaw_dumbbell > -73.89
##                             384) gyros_forearm_z <= 0.57; criterion = 1, statistic = 26.418
##                               385) yaw_forearm <= -87.7; criterion = 0.999, statistic = 24.233
##                                 386)*  weights = 7 
##                               385) yaw_forearm > -87.7
##                                 387)*  weights = 37 
##                             384) gyros_forearm_z > 0.57
##                               388)*  weights = 18 
##                     345) pitch_forearm > 59.3
##                       389) magnet_arm_y <= -2; criterion = 1, statistic = 30.92
##                         390)*  weights = 19 
##                       389) magnet_arm_y > -2
##                         391)*  weights = 42 
##               282) magnet_belt_x > 32
##                 392) magnet_dumbbell_z <= -31; criterion = 1, statistic = 26.934
##                   393)*  weights = 24 
##                 392) magnet_dumbbell_z > -31
##                   394)*  weights = 19 
##             281) total_accel_dumbbell > 11
##               395) accel_dumbbell_z <= 103; criterion = 1, statistic = 43.811
##                 396) gyros_belt_z <= -0.44; criterion = 1, statistic = 31.489
##                   397)*  weights = 14 
##                 396) gyros_belt_z > -0.44
##                   398) magnet_belt_y <= 586; criterion = 0.996, statistic = 18.712
##                     399)*  weights = 9 
##                   398) magnet_belt_y > 586
##                     400)*  weights = 67 
##               395) accel_dumbbell_z > 103
##                 401)*  weights = 7 
##           280) total_accel_dumbbell > 21
##             402) magnet_forearm_x <= -515; criterion = 1, statistic = 57.489
##               403) magnet_dumbbell_x <= -419; criterion = 1, statistic = 42.594
##                 404)*  weights = 86 
##               403) magnet_dumbbell_x > -419
##                 405)*  weights = 7 
##             402) magnet_forearm_x > -515
##               406)*  weights = 10 
##       73) magnet_dumbbell_z > 48
##         407) yaw_dumbbell <= -70.37; criterion = 1, statistic = 576.03
##           408) pitch_belt <= -43.2; criterion = 1, statistic = 238.169
##             409) pitch_belt <= -45.1; criterion = 1, statistic = 123.764
##               410) magnet_forearm_x <= -428; criterion = 1, statistic = 30.765
##                 411)*  weights = 35 
##               410) magnet_forearm_x > -428
##                 412)*  weights = 8 
##             409) pitch_belt > -45.1
##               413) roll_belt <= 121; criterion = 1, statistic = 125.625
##                 414)*  weights = 23 
##               413) roll_belt > 121
##                 415) magnet_belt_z <= -334; criterion = 1, statistic = 102.687
##                   416)*  weights = 11 
##                 415) magnet_belt_z > -334
##                   417) accel_belt_x <= 46; criterion = 1, statistic = 78.996
##                     418)*  weights = 19 
##                   417) accel_belt_x > 46
##                     419) accel_belt_x <= 48; criterion = 1, statistic = 83.699
##                       420)*  weights = 18 
##                     419) accel_belt_x > 48
##                       421) accel_belt_x <= 53; criterion = 1, statistic = 80.252
##                         422) roll_belt <= 125; criterion = 0.999, statistic = 18.548
##                           423)*  weights = 18 
##                         422) roll_belt > 125
##                           424)*  weights = 82 
##                       421) accel_belt_x > 53
##                         425)*  weights = 10 
##           408) pitch_belt > -43.2
##             426) magnet_dumbbell_y <= 296; criterion = 1, statistic = 216.646
##               427) yaw_arm <= -126; criterion = 1, statistic = 126.725
##                 428)*  weights = 35 
##               427) yaw_arm > -126
##                 429) roll_arm <= -88.1; criterion = 1, statistic = 64.321
##                   430)*  weights = 10 
##                 429) roll_arm > -88.1
##                   431) gyros_belt_x <= 0.22; criterion = 1, statistic = 60.218
##                     432) accel_dumbbell_y <= 86; criterion = 1, statistic = 43.486
##                       433) magnet_arm_z <= -345; criterion = 1, statistic = 66.073
##                         434)*  weights = 7 
##                       433) magnet_arm_z > -345
##                         435) gyros_dumbbell_y <= 0.34; criterion = 1, statistic = 138.798
##                           436) magnet_belt_y <= 611; criterion = 1, statistic = 54.228
##                             437) gyros_belt_x <= -0.11; criterion = 1, statistic = 27.983
##                               438)*  weights = 7 
##                             437) gyros_belt_x > -0.11
##                               439) yaw_arm <= 122; criterion = 0.986, statistic = 16.391
##                                 440)*  weights = 291 
##                               439) yaw_arm > 122
##                                 441)*  weights = 9 
##                           436) magnet_belt_y > 611
##                             442) magnet_dumbbell_y <= 214; criterion = 0.987, statistic = 13.406
##                               443)*  weights = 21 
##                             442) magnet_dumbbell_y > 214
##                               444)*  weights = 12 
##                         435) gyros_dumbbell_y > 0.34
##                           445)*  weights = 8 
##                     432) accel_dumbbell_y > 86
##                       446) accel_forearm_x <= -78; criterion = 1, statistic = 34.795
##                         447) roll_arm <= -15.4; criterion = 1, statistic = 48.352
##                           448)*  weights = 11 
##                         447) roll_arm > -15.4
##                           449) gyros_belt_y <= -0.02; criterion = 1, statistic = 61.645
##                             450)*  weights = 9 
##                           449) gyros_belt_y > -0.02
##                             451) yaw_forearm <= 125; criterion = 1, statistic = 19.966
##                               452)*  weights = 25 
##                             451) yaw_forearm > 125
##                               453) yaw_belt <= -87.7; criterion = 0.999, statistic = 18.392
##                                 454)*  weights = 49 
##                               453) yaw_belt > -87.7
##                                 455)*  weights = 8 
##                       446) accel_forearm_x > -78
##                         456) yaw_arm <= -35; criterion = 1, statistic = 43.142
##                           457)*  weights = 7 
##                         456) yaw_arm > -35
##                           458) magnet_forearm_y <= 298; criterion = 1, statistic = 27.112
##                             459)*  weights = 10 
##                           458) magnet_forearm_y > 298
##                             460) magnet_forearm_z <= 811; criterion = 1, statistic = 29.113
##                               461) magnet_belt_z <= -316; criterion = 1, statistic = 23.129
##                                 462)*  weights = 7 
##                               461) magnet_belt_z > -316
##                                 463)*  weights = 109 
##                             460) magnet_forearm_z > 811
##                               464) accel_forearm_z <= -155; criterion = 0.997, statistic = 19.301
##                                 465)*  weights = 14 
##                               464) accel_forearm_z > -155
##                                 466)*  weights = 11 
##                   431) gyros_belt_x > 0.22
##                     467)*  weights = 18 
##             426) magnet_dumbbell_y > 296
##               468) accel_forearm_x <= -76; criterion = 1, statistic = 159.313
##                 469) accel_arm_x <= -140; criterion = 1, statistic = 79.132
##                   470) gyros_belt_y <= -0.02; criterion = 1, statistic = 32.793
##                     471)*  weights = 9 
##                   470) gyros_belt_y > -0.02
##                     472)*  weights = 30 
##                 469) accel_arm_x > -140
##                   473) magnet_belt_z <= -345; criterion = 1, statistic = 48.627
##                     474)*  weights = 18 
##                   473) magnet_belt_z > -345
##                     475) magnet_dumbbell_y <= 338; criterion = 0.999, statistic = 22.482
##                       476) gyros_belt_y <= 0.02; criterion = 1, statistic = 23.124
##                         477) gyros_arm_z <= -0.18; criterion = 0.996, statistic = 19.033
##                           478)*  weights = 7 
##                         477) gyros_arm_z > -0.18
##                           479)*  weights = 26 
##                       476) gyros_belt_y > 0.02
##                         480) roll_arm <= -16.4; criterion = 1, statistic = 21.158
##                           481)*  weights = 7 
##                         480) roll_arm > -16.4
##                           482)*  weights = 27 
##                     475) magnet_dumbbell_y > 338
##                       483)*  weights = 122 
##               468) accel_forearm_x > -76
##                 484) accel_dumbbell_z <= -143; criterion = 1, statistic = 105.751
##                   485) accel_arm_x <= -43; criterion = 1, statistic = 44.416
##                     486) magnet_forearm_y <= 32; criterion = 0.998, statistic = 23.173
##                       487)*  weights = 9 
##                     486) magnet_forearm_y > 32
##                       488) gyros_dumbbell_y <= 0.51; criterion = 0.992, statistic = 20.294
##                         489) gyros_belt_y <= 0; criterion = 1, statistic = 41.163
##                           490) gyros_dumbbell_y <= -0.03; criterion = 0.968, statistic = 14.786
##                             491)*  weights = 8 
##                           490) gyros_dumbbell_y > -0.03
##                             492)*  weights = 26 
##                         489) gyros_belt_y > 0
##                           493) magnet_arm_z <= 416; criterion = 0.997, statistic = 19.35
##                             494)*  weights = 8 
##                           493) magnet_arm_z > 416
##                             495)*  weights = 16 
##                       488) gyros_dumbbell_y > 0.51
##                         496)*  weights = 8 
##                   485) accel_arm_x > -43
##                     497) magnet_dumbbell_x <= -462; criterion = 1, statistic = 66.283
##                       498) accel_forearm_z <= -161; criterion = 0.997, statistic = 19.607
##                         499)*  weights = 41 
##                       498) accel_forearm_z > -161
##                         500) roll_forearm <= 134; criterion = 1, statistic = 26.047
##                           501)*  weights = 23 
##                         500) roll_forearm > 134
##                           502)*  weights = 15 
##                     497) magnet_dumbbell_x > -462
##                       503)*  weights = 7 
##                 484) accel_dumbbell_z > -143
##                   504) pitch_belt <= -41.7; criterion = 0.999, statistic = 27.752
##                     505) pitch_belt <= -42; criterion = 0.995, statistic = 18.657
##                       506)*  weights = 12 
##                     505) pitch_belt > -42
##                       507)*  weights = 10 
##                   504) pitch_belt > -41.7
##                     508)*  weights = 7 
##         407) yaw_dumbbell > -70.37
##           509) magnet_belt_y <= 623; criterion = 1, statistic = 405.537
##             510) total_accel_dumbbell <= 3; criterion = 1, statistic = 333.411
##               511) gyros_belt_z <= 0.07; criterion = 1, statistic = 151.742
##                 512) magnet_dumbbell_y <= 221; criterion = 1, statistic = 103.456
##                   513) accel_belt_x <= -8; criterion = 1, statistic = 27.632
##                     514)*  weights = 8 
##                   513) accel_belt_x > -8
##                     515)*  weights = 33 
##                 512) magnet_dumbbell_y > 221
##                   516) yaw_belt <= -88.4; criterion = 1, statistic = 72
##                     517)*  weights = 11 
##                   516) yaw_belt > -88.4
##                     518)*  weights = 96 
##               511) gyros_belt_z > 0.07
##                 519)*  weights = 8 
##             510) total_accel_dumbbell > 3
##               520) accel_belt_y <= -2; criterion = 1, statistic = 153.72
##                 521) magnet_dumbbell_z <= 330; criterion = 1, statistic = 52.274
##                   522)*  weights = 58 
##                 521) magnet_dumbbell_z > 330
##                   523) pitch_dumbbell <= 36.67; criterion = 0.999, statistic = 21.723
##                     524)*  weights = 20 
##                   523) pitch_dumbbell > 36.67
##                     525)*  weights = 7 
##               520) accel_belt_y > -2
##                 526) accel_forearm_x <= 195; criterion = 1, statistic = 151.071
##                   527) accel_forearm_x <= 94; criterion = 1, statistic = 95.229
##                     528) yaw_arm <= -40.6; criterion = 1, statistic = 57.33
##                       529) roll_dumbbell <= -49.96; criterion = 1, statistic = 99.951
##                         530)*  weights = 12 
##                       529) roll_dumbbell > -49.96
##                         531) roll_dumbbell <= 50.5; criterion = 1, statistic = 72.046
##                           532) gyros_forearm_x <= 0; criterion = 0.999, statistic = 25.617
##                             533) gyros_forearm_x <= -0.35; criterion = 0.999, statistic = 21.369
##                               534)*  weights = 7 
##                             533) gyros_forearm_x > -0.35
##                               535)*  weights = 17 
##                           532) gyros_forearm_x > 0
##                             536)*  weights = 7 
##                         531) roll_dumbbell > 50.5
##                           537) gyros_arm_x <= -2.86; criterion = 1, statistic = 28.148
##                             538)*  weights = 16 
##                           537) gyros_arm_x > -2.86
##                             539)*  weights = 80 
##                     528) yaw_arm > -40.6
##                       540) pitch_forearm <= 33; criterion = 1, statistic = 111.852
##                         541) accel_belt_x <= -17; criterion = 1, statistic = 92.228
##                           542)*  weights = 19 
##                         541) accel_belt_x > -17
##                           543) gyros_belt_y <= -0.05; criterion = 1, statistic = 84.866
##                             544)*  weights = 7 
##                           543) gyros_belt_y > -0.05
##                             545) yaw_dumbbell <= 68.57; criterion = 1, statistic = 81.948
##                               546) total_accel_dumbbell <= 27; criterion = 1, statistic = 81.382
##                                 547) total_accel_dumbbell <= 5; criterion = 1, statistic = 69.532
##                                   548)*  weights = 28 
##                                 547) total_accel_dumbbell > 5
##                                   549) accel_forearm_z <= -159; criterion = 1, statistic = 51.201
##                                     550)*  weights = 7 
##                                   549) accel_forearm_z > -159
##                                     551) gyros_belt_z <= -0.36; criterion = 1, statistic = 84.32
##                                       552)*  weights = 7 
##                                     551) gyros_belt_z > -0.36
##                                       553) magnet_forearm_z <= -566; criterion = 1, statistic = 47.646
##                                         554)*  weights = 8 
##                                       553) magnet_forearm_z > -566
##                                         555) yaw_arm <= -10; criterion = 1, statistic = 35.248
##                                           556) magnet_arm_y <= 248; criterion = 1, statistic = 24.017
##                                             557)*  weights = 44 
##                                           556) magnet_arm_y > 248
##                                             558) pitch_belt <= -43.8; criterion = 1, statistic = 26.636
##                                               559)*  weights = 18 
##                                             558) pitch_belt > -43.8
##                                               560) roll_belt <= 125; criterion = 1, statistic = 20.983
##                                                 561)*  weights = 26 
##                                               560) roll_belt > 125
##                                                 562)*  weights = 8 
##                                         555) yaw_arm > -10
##                                           563) total_accel_dumbbell <= 16; criterion = 1, statistic = 24.45
##                                             564)*  weights = 7 
##                                           563) total_accel_dumbbell > 16
##                                             565)*  weights = 158 
##                               546) total_accel_dumbbell > 27
##                                 566)*  weights = 12 
##                             545) yaw_dumbbell > 68.57
##                               567) roll_belt <= 0.2; criterion = 1, statistic = 135.228
##                                 568)*  weights = 7 
##                               567) roll_belt > 0.2
##                                 569) yaw_belt <= -88.1; criterion = 1, statistic = 71.305
##                                   570) magnet_dumbbell_y <= 275; criterion = 0.997, statistic = 19.327
##                                     571) accel_dumbbell_y <= 15; criterion = 0.999, statistic = 19.23
##                                       572)*  weights = 14 
##                                     571) accel_dumbbell_y > 15
##                                       573)*  weights = 9 
##                                   570) magnet_dumbbell_y > 275
##                                     574) accel_arm_z <= -221; criterion = 0.967, statistic = 11.632
##                                       575)*  weights = 7 
##                                     574) accel_arm_z > -221
##                                       576) gyros_forearm_x <= 0.24; criterion = 0.973, statistic = 11.99
##                                         577)*  weights = 60 
##                                       576) gyros_forearm_x > 0.24
##                                         578)*  weights = 7 
##                                 569) yaw_belt > -88.1
##                                   579)*  weights = 57 
##                       540) pitch_forearm > 33
##                         580) roll_dumbbell <= 34.93; criterion = 1, statistic = 34.235
##                           581)*  weights = 11 
##                         580) roll_dumbbell > 34.93
##                           582) accel_dumbbell_z <= 34; criterion = 1, statistic = 28.632
##                             583)*  weights = 9 
##                           582) accel_dumbbell_z > 34
##                             584)*  weights = 36 
##                   527) accel_forearm_x > 94
##                     585) total_accel_dumbbell <= 7; criterion = 1, statistic = 66.546
##                       586) gyros_forearm_y <= -4; criterion = 1, statistic = 42.302
##                         587)*  weights = 8 
##                       586) gyros_forearm_y > -4
##                         588) accel_forearm_z <= 137; criterion = 1, statistic = 43.013
##                           589)*  weights = 8 
##                         588) accel_forearm_z > 137
##                           590) gyros_forearm_y <= -1.03; criterion = 1, statistic = 31.964
##                             591) pitch_belt <= 4.32; criterion = 0.999, statistic = 22.256
##                               592) accel_belt_x <= -8; criterion = 0.959, statistic = 11.214
##                                 593)*  weights = 8 
##                               592) accel_belt_x > -8
##                                 594)*  weights = 17 
##                             591) pitch_belt > 4.32
##                               595)*  weights = 7 
##                           590) gyros_forearm_y > -1.03
##                             596)*  weights = 7 
##                     585) total_accel_dumbbell > 7
##                       597) roll_dumbbell <= 39.33; criterion = 1, statistic = 43.95
##                         598) roll_forearm <= 145; criterion = 0.992, statistic = 14.337
##                           599)*  weights = 7 
##                         598) roll_forearm > 145
##                           600)*  weights = 33 
##                       597) roll_dumbbell > 39.33
##                         601) magnet_belt_y <= 580; criterion = 1, statistic = 25.255
##                           602) accel_dumbbell_z <= 66; criterion = 1, statistic = 32.064
##                             603)*  weights = 7 
##                           602) accel_dumbbell_z > 66
##                             604) gyros_dumbbell_z <= -0.2; criterion = 0.955, statistic = 11.04
##                               605)*  weights = 8 
##                             604) gyros_dumbbell_z > -0.2
##                               606)*  weights = 34 
##                         601) magnet_belt_y > 580
##                           607)*  weights = 7 
##                 526) accel_forearm_x > 195
##                   608) pitch_forearm <= -0.53; criterion = 1, statistic = 71.241
##                     609)*  weights = 8 
##                   608) pitch_forearm > -0.53
##                     610) yaw_dumbbell <= 57.32; criterion = 1, statistic = 61.93
##                       611)*  weights = 16 
##                     610) yaw_dumbbell > 57.32
##                       612) magnet_dumbbell_z <= 397; criterion = 0.983, statistic = 12.897
##                         613)*  weights = 86 
##                       612) magnet_dumbbell_z > 397
##                         614)*  weights = 7 
##           509) magnet_belt_y > 623
##             615) accel_forearm_x <= -102; criterion = 1, statistic = 181.873
##               616) magnet_arm_x <= -263; criterion = 1, statistic = 29.29
##                 617)*  weights = 7 
##               616) magnet_arm_x > -263
##                 618)*  weights = 98 
##             615) accel_forearm_x > -102
##               619) roll_belt <= -14.1; criterion = 1, statistic = 268.012
##                 620)*  weights = 58 
##               619) roll_belt > -14.1
##                 621) roll_belt <= -0.84; criterion = 1, statistic = 166.311
##                   622)*  weights = 83 
##                 621) roll_belt > -0.84
##                   623) pitch_forearm <= -1.77; criterion = 1, statistic = 79.283
##                     624) roll_forearm <= 138; criterion = 1, statistic = 41.151
##                       625)*  weights = 28 
##                     624) roll_forearm > 138
##                       626)*  weights = 19 
##                   623) pitch_forearm > -1.77
##                     627) magnet_belt_z <= -308; criterion = 1, statistic = 23.525
##                       628)*  weights = 77 
##                     627) magnet_belt_z > -308
##                       629) roll_belt <= -0.03; criterion = 0.997, statistic = 19.236
##                         630)*  weights = 27 
##                       629) roll_belt > -0.03
##                         631) magnet_dumbbell_z <= 117; criterion = 0.996, statistic = 19.079
##                           632)*  weights = 22 
##                         631) magnet_dumbbell_z > 117
##                           633)*  weights = 19
```

Model consisted of 317 nodes and 633 level depth and was not easily interpretable. 

After fitting model and cross validating on the training set result produced was fairly good with accuracy of 85%.



```r
ctClass <- predict(ctFit, newdata = testing)
confusionMatrix(ctClass, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2505  117   32   47   35
##          B  112 1576  126   66   81
##          C   40   78 1360  133   76
##          D   52   43   99 1272   52
##          E   26   45   59   55 1520
## 
## Overall Statistics
##                                        
##                Accuracy : 0.857        
##                  95% CI : (0.85, 0.864)
##     No Information Rate : 0.285        
##     P-Value [Acc > NIR] : < 2e-16      
##                                        
##                   Kappa : 0.819        
##  Mcnemar's Test P-Value : 7.72e-05     
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.916    0.848    0.811    0.809    0.862
## Specificity             0.966    0.950    0.959    0.969    0.976
## Pos Pred Value          0.916    0.804    0.806    0.838    0.891
## Neg Pred Value          0.967    0.963    0.960    0.963    0.969
## Prevalence              0.285    0.194    0.174    0.164    0.184
## Detection Rate          0.261    0.164    0.142    0.132    0.158
## Detection Prevalence    0.285    0.204    0.176    0.158    0.177
## Balanced Accuracy       0.941    0.899    0.885    0.889    0.919
```

In conclusion rpart and ctree methods accuracy was lower and there was no improvement in readability of the model. Random Forest model still offered optimal accuracy compared to other models.
