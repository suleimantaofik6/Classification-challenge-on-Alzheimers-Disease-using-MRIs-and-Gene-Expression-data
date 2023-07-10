rm(list=ls())

library(caret)
library(pROC)
library(mltools)
library(magrittr)
library(dplyr)

# ================================================================================
# CODE 1: MARKER FOR ADCTL
# ================================================================================

# Load the RESULT dataset submitted
ADCTLresult <- read.csv("C:/Users/sulei/Downloads/Tutorials/R/SL ML challenge 1/0075720_Suleiman_ADCTLres.csv")
prediction1 <- ADCTLresult[, "Prediction"]

# Load the test dataset and extract the real labels
ADCTLtest_wl <- read.csv("C:/Users/sulei/Downloads/Tutorials/R/SL ML challenge 1/ADCTLtest_wl.csv")
ADCTLtest_label <- ADCTLtest_wl[, "Label"]

# Convert real test label to 0 and 1 and make it a factor
ADCTLtest_label <- ifelse(ADCTLtest_label == "AD", 1, 0)
ADCTLtest_label <- factor(ADCTLtest_label)

#================================================================================
# Compute performance metrics FOR ADCTL
#================================================================================

TP1 <- sum(prediction1 == 1 & ADCTLtest_label== 1)
TN1 <- sum(prediction1 == 0 & ADCTLtest_label == 0)
FP1 <- sum(prediction1 == 1 & ADCTLtest_label == 0)
FN1 <- sum(prediction1 == 0 & ADCTLtest_label == 1)

accuracy1 <- (TP1 + TN1) / (TP1 + TN1 + FP1 + FN1)
sensitivity1 <- TP1 / (TP1 + FN1)
specificity1 <- TN1 / (TN1 + FP1)
precision1 <- TP1 / (TP1 + FP1)
f1_score1 <- 2 * (precision1 * sensitivity1) / (precision1 + sensitivity1)
auc1 <- roc(ADCTLtest_label, as.numeric(prediction1))$auc
MCC1 <- (TP1 * TN1 - FP1 * FN1) / sqrt((TP1 + FP1) * (TP1 + FN1) * (TN1 + FP1) * (TN1 + FN1))
balanced_accuracy1 <- (sensitivity1 + specificity1) / 2

# Convert labels and predictions to factors
label1 <- factor(as.character(ADCTLtest_label), levels = c("0", "1"))
predictions1 <- factor(as.character(prediction1), levels = c("0", "1"))

# Compute ROC and AUC
roc_obj1 <- roc(label1, as.numeric(predictions1))
auc1 <- auc(roc_obj1)

# Create a data frame with the metrics
metricsADCTL <- data.frame(Accuracy = accuracy1,
                       Sensitivity = sensitivity1,
                       Specificity = specificity1,
                       Precision = precision1,
                       F1_Score = f1_score1,
                       AUC = auc1,
                       MCC = MCC1,
                       Balanced_Accuracy = balanced_accuracy1)

# Print the metrics table
print(metricsADCTL)



# ================================================================================
# CODE 2: MARKER FOR MCICTL
# ================================================================================

# Load the RESULT dataset submitted
MCICTLresult <- read.csv("C:/Users/sulei/Downloads/Tutorials/R/SL ML challenge 1/0075720_Suleiman_MCICTLres.csv")
prediction2 <- MCICTLresult[, "Prediction"]

# Load the test dataset and extract the real labels
MCICTLtest <- read.csv("C:/Users/sulei/Downloads/Tutorials/R/SL ML challenge 1/MCICTLtest.csv")
MCICTLtest_wl <- read.csv("C:/Users/sulei/Downloads/Tutorials/R/SL ML challenge 1/MCICTLtest_wl.csv")
MCICTLtest_label <- MCICTLtest_wl[, "Label"]

# Convert real test label to 0 and 1 and make it a factor
MCICTLtest_label <- ifelse(MCICTLtest_label == "MCI", 1, 0)
MCICTLtest_label <- factor(MCICTLtest_label)


#================================================================================
# Compute performance metrics on the TEST data
#================================================================================

TP2 <- sum(prediction2 == 1 & MCICTLtest_label== 1)
TN2 <- sum(prediction2 == 0 & MCICTLtest_label == 0)
FP2 <- sum(prediction2 == 1 & MCICTLtest_label == 0)
FN2 <- sum(prediction2 == 0 & MCICTLtest_label== 1)

accuracy2 <- (TP2 + TN2) / (TP2 + TN2 + FP2 + FN2)
sensitivity2 <- TP2 / (TP2 + FN2)
specificity2 <- TN2 / (TN2 + FP2)
precision2 <- TP2 / (TP2 + FP2)
f1_score2 <- 2 * (precision2 * sensitivity2) / (precision2 + sensitivity2)
auc2 <- roc(MCICTLtest_label, as.numeric(prediction2))$auc
MCC2 <- (TP2 * TN2 - FP2 * FN2) / sqrt((TP2 + FP2) * (TP2 + FN2) * (TN2 + FP2) * (TN2 + FN2))
balanced_accuracy2 <- (sensitivity2 + specificity2) / 2

# Convert labels and predictions to factors
label2 <- factor(as.character(MCICTLtest_label), levels = c("0", "1"))
predictions2 <- factor(as.character(prediction2), levels = c("0", "1"))

# Compute ROC and AUC
roc_obj2 <- roc(label2, as.numeric(predictions2))
auc2 <- auc(roc_obj2)

# Create a data frame with the metrics
metricsMCICTL <- data.frame(Accuracy = accuracy2,
                       Sensitivity = sensitivity2,
                       Specificity = specificity2,
                       Precision = precision2,
                       F1_Score = f1_score2,
                       AUC = auc2,
                       MCC = MCC2,
                       Balanced_Accuracy = balanced_accuracy2)

# Print the metrics table
print(metricsMCICTL)



# ================================================================================
# CODE 3: MARKER FOR ADMCI
# ================================================================================

# Load the RESULT dataset submitted
ADMCIresult <- read.csv("C:/Users/sulei/Downloads/Tutorials/R/SL ML challenge 1/0075720_Suleiman_ADMCIres.csv")
prediction3 <- ADMCIresult[, "Prediction"]


# Load the test dataset and extract the real labels
ADMCItest <- read.csv("C:/Users/sulei/Downloads/Tutorials/R/SL ML challenge 1/ADMCItest.csv")
ADMCItest_wl <- read.csv("C:/Users/sulei/Downloads/Tutorials/R/SL ML challenge 1/ADMCItest_wl.csv")
ADMCI_label <- ADMCItest_wl[, "Label"]
# Convert real test label to 0 and 1 and make it a factor
ADMCI_label <- ifelse(ADMCI_label == "AD", 1, 0)
ADMCI_label <- factor(ADMCI_label)


#================================================================================
# Compute performance metrics on the TEST data
#================================================================================

TP3 <- sum(prediction3 == 1 & ADMCI_label == 1)
TN3 <- sum(prediction3 == 0 & ADMCI_label == 0)
FP3 <- sum(prediction3 == 1 & ADMCI_label == 0)
FN3 <- sum(prediction3 == 0 & ADMCI_label == 1)

accuracy3 <- (TP3 + TN3) / (TP3 + TN3 + FP3 + FN3)
sensitivity3 <- TP3 / (TP3 + FN3)
specificity3 <- TN3 / (TN3 + FP3)
precision3 <- TP3 / (TP3 + FP3)
f1_score3 <- 2 * (precision3 * sensitivity3) / (precision3 + sensitivity3)
auc3 <- roc(ADMCI_label, as.numeric(prediction3))$auc
MCC3 <- (TP3 * TN3 - FP3 * FN3) / sqrt((TP3 + FP3) * (TP3 + FN3) * (TN3 + FP3) * (TN3 + FN3))
balanced_accuracy3 <- (sensitivity3 + specificity3) / 2

# Convert labels and predictions to factors
label3 <- factor(as.character(ADMCI_label), levels = c("0", "1"))
predictions3 <- factor(as.character(prediction3), levels = c("0", "1"))

# Compute ROC and AUC
roc_obj3 <- roc(label3, as.numeric(predictions3))
auc3 <- auc(roc_obj3)

# Create a data frame with the metrics
metricsADMCI <- data.frame(Accuracy = accuracy3,
                       Sensitivity = sensitivity3,
                       Specificity = specificity3,
                       Precision = precision3,
                       F1_Score = f1_score3,
                       AUC = auc3,
                       MCC = MCC3,
                       Balanced_Accuracy = balanced_accuracy3)

# Print the metrics table
print(metricsADCTL)
print(metricsMCICTL)
print(metricsADMCI)
