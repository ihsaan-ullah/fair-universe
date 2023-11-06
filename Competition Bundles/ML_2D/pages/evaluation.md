# Evaluation

The problem is a binary (two classes) classification problem. Each sample (a 2D point) is characterized by its coordinates x1 and x2 (2 features). You must predict the points category: signal or background.


### TO DO:
You are given for training a data matrix X_train of dimension num_training_samples x 2 (2 features, x1 and x2) and an array y_train of labels of dimension num_training_samples. You must train a model which predicts the labels for two test matrices X_valid and X_test.


### Phases: 
**Phase 1** - Development phase: We provide you with labeled training data and unlabeled validation and test data. Make predictions for both datasets. However, you will receive feed-back on your performance on the validation set only. The performance of your LAST submission will be displayed on the leaderboard.

**Phase 2** - Final phase. You do not need to do anything. Your last submission of phase 1 will be automatically forwarded. Your performance on the test set will appear on the leaderboard when the organizers finish checking the submissions.


### Submissions
This competition allows you to submit either:
- Only prediction results (results submission)
- A pre-trained prediction model (code submission)
- A prediction model that must be trained and tested (code submission)

### Metric
The submissions are evaluated using the `ROC AUC metric`.  

**ROC AUC**  
A ROC (Receiver Operating Characteristic) curve plots the performance of a classification model (considering two parameters, the True Positive Rate and the False Positive Rate) at different classification thresholds. Then, the ROC AUC stands for "Area under the ROC Curve": AUC ROC measures the entire two-dimensional area underneath the entire ROC curve from (0,0) to (1,1). ROC AUC provides an aggregate measure of performance across all possible classification thresholds: one way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example. AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.

***

To know how AUC works, check this blog post: https://arize.com/blog/what-is-auc

AUC is calcualted using scikit-learn : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
