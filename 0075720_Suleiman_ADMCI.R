rm(list=ls())

library(caret)
library(pROC)
library(mltools)
library(magrittr)
library(dplyr)

# ==============================================================================
# Load the training dataset and drop the ID column
ADMCI <- read.csv("C:/Users/sulei/Downloads/Tutorials/R/SL ML challenge 1/ADMCItrain.csv")
str(ADMCI)
ADMCI <- ADMCI[-1]

# Convert label to 0 and 1 and make it a factor
ADMCI$Label <- ifelse(ADMCI$Label == "AD", 1, 0)
ADMCI$Label <- factor(ADMCI$Label)

# Apply cross-validation
set.seed(3033)
trainsplit <- createDataPartition(y = ADMCI$Label, p = 0.7, list = FALSE)
training <- ADMCI[trainsplit, ]
testing <- ADMCI[-trainsplit, ]

# Normalize the features
preproc_range <- preProcess(training[, -ncol(training)], 
                            method = "range", range = c(0, 1))
training_norm <- predict(preproc_range, training)
testing_norm0 <- predict(preproc_range, testing)

# Apply PCA on the normalized data
preproc_pca <- preProcess(training_norm[, -ncol(training_norm)], 
                          method = "pca", pcaComp = 5)
training_pca <- predict(preproc_pca, training_norm)
testing_pca0 <- predict(preproc_pca, testing_norm0)

# Extract the column detected by the PCA from the rotation matrix
pca_rot_mat <- preproc_pca$rotation
pca_columns <- head(order(-abs(pca_rot_mat[, 1])), 5)
feature_used <- colnames(training_norm)[pca_columns]

# Set up train control and build the logistic regression model
trainC <- trainControl(method = 'repeatedcv', number = 5, repeats = 3)
logreg <- train(Label ~ ., data = training_pca, method = 'glm', 
                trControl = trainC, tuneLength = 10, family = "binomial")

# ==============================================================================

# Compute performance metrics on the splitted data
prediction <- predict(logreg, newdata = testing_pca0)

TP <- sum(prediction == 1 & testing_pca0$Label == 1)
TN <- sum(prediction == 0 & testing_pca0$Label == 0)
FP <- sum(prediction == 1 & testing_pca0$Label == 0)
FN <- sum(prediction == 0 & testing_pca0$Label == 1)

accuracy <- (TP + TN) / (TP + TN + FP + FN)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
precision <- TP / (TP + FP)
f1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)
auc <- roc(testing_pca0$Label, as.numeric(prediction))$auc
MCC <- (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
balanced_accuracy <- (sensitivity + specificity) / 2

# Convert labels and predictions to factors
labels <- factor(as.character(testing_pca0$Label), levels = c("0", "1"))
predictions <- factor(as.character(prediction), levels = c("0", "1"))

# Compute ROC and AUC
roc_obj <- roc(labels, as.numeric(predictions))
auc <- auc(roc_obj)

# Create a data frame with the metrics
metrics <- data.frame(Accuracy = accuracy,
                      Sensitivity = sensitivity,
                      Specificity = specificity,
                      Precision = precision,
                      F1_Score = f1_score,
                      AUC = auc,
                      MCC = MCC,
                      Balanced_Accuracy = balanced_accuracy)

# Print the metrics table
print(metrics)

# ==============================================================================
# TIME TO TEST THE MODEL ON THE REAL TEST DATA
# ==============================================================================

# Load the test dataset, process it with the fetures used in the train and predict
ADMCItest <- read.csv("C:/Users/sulei/Downloads/Tutorials/R/SL ML challenge 1/ADMCItest.csv")
ADMCItest_wl <- read.csv("C:/Users/sulei/Downloads/Tutorials/R/SL ML challenge 1/ADMCItest_wl.csv")
ADMCI_label <- ADMCItest_wl[, "Label"]
# Convert real test label to 0 and 1 and make it a factor
ADMCI_label <- ifelse(ADMCI_label == "AD", 1, 0)
ADMCI_label <- factor(ADMCI_label)


test_ids <- ADMCItest$ID
ADMCItest <- ADMCItest[-1]
testing_norm <- predict(preproc_range, ADMCItest)
testing_pca <- predict(preproc_pca, testing_norm)

# Predict on the test set using the trained model
prediction1 <- predict(logreg, newdata = testing_pca)
#probabilities <- prediction1[, "1"]
#probabilities <- round(probabilities, 4)

# Create a data frame with the ID, predicted labels, and probabilities
#results <- data.frame(ID = test_ids, 
                      #Prediction = ifelse(probabilities > 0.5, 1, 0),
                      #Prob_0 = 1 - probabilities,
                      #Prob_1 = probabilities)

#================================================================================
# Compute performance metrics on the TEST data
#================================================================================

TP1 <- sum(prediction1 == 1 & ADMCI_label == 1)
TN1 <- sum(prediction1 == 0 & ADMCI_label == 0)
FP1 <- sum(prediction1 == 1 & ADMCI_label == 0)
FN1 <- sum(prediction1 == 0 & ADMCI_label == 1)

accuracy1 <- (TP1 + TN1) / (TP1 + TN1 + FP1 + FN1)
sensitivity1 <- TP1 / (TP + FN)
specificity1 <- TN1 / (TN1 + FP1)
precision1 <- TP1 / (TP1 + FP1)
f1_score1 <- 2 * (precision1 * sensitivity1) / (precision1 + sensitivity1)
auc1 <- roc(ADMCI_label, as.numeric(prediction1))$auc
MCC1 <- (TP1 * TN1 - FP1 * FN1) / sqrt((TP1 + FP1) * (TP1 + FN1) * (TN1 + FP1) * (TN1 + FN1))
balanced_accuracy1 <- (sensitivity1 + specificity1) / 2

# Convert labels and predictions to factors
label <- factor(as.character(ADMCI_label), levels = c("0", "1"))
predictions1 <- factor(as.character(prediction1), levels = c("0", "1"))

# Compute ROC and AUC
roc_obj1 <- roc(label, as.numeric(predictions1))
auc <- auc(roc_obj)

# Create a data frame with the metrics
metrics1 <- data.frame(Accuracy = accuracy1,
                      Sensitivity = sensitivity1,
                      Specificity = specificity1,
                      Precision = precision1,
                      F1_Score = f1_score1,
                      AUC = auc1,
                      MCC = MCC1,
                      Balanced_Accuracy = balanced_accuracy1)

# Print the metrics table
print(metrics1)

# Save the results to a CSV file
# write.csv(results, "C:/Users/sulei/Downloads/Tutorials/R/SL ML challenge 1/0075720_Suleiman_ADMCIres.csv", row.names = FALSE)

# Save the pca_column features to a CSV file
# write.csv(feature_used, "C:/Users/sulei/Downloads/Tutorials/R/SL ML challenge 1/0075720_Suleiman_MCICTLfeat.csv", row.names = FALSE)