# Classification-challenge-on-Alzheimers-Disease-using-MRIs-and-Gene-Expression-data

Alzheimer’s disease is a major public health concern affecting millions of people worldwide, consequently, early diagnosis is crucial for effective treatment
In this classification challenge, I used MRIs and gene expression data to classify patients into three macro-stages of Alzheimer’s disease: 
1. CTL (Controls): No deficit
2. MCI (Mild Cognitive Impairment): Few deficits
3. AD (Alzheimer’s Disease): Dementia

For this challenge, a pair of three datasets were used. This consists of three train datasets with each corresponding to their equivalent test data.
The summary of the datasets and their respective feature sizes are:
1. ADCTLtrain = 430 features + 1 label
2. ADMCItrain = 64 features + 1 label
3. MCICTLtrain = 594 features + 1 label

These features include data from demographic, clinical, CSF, medical imaging, and transcriptomics. With these large features there is a need to find optimal solutions to three classification problems:
AD vs CTL
AD vs MCI
MCI VS CTL
