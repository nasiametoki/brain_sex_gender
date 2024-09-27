# brain_sex_gender

This is the analysis workflow for:

## Overview
* ## SVM
  
  * 1_SVM_rsfc_sex.m : Support Vector Machine classification. Model training using the resting-state functional connectivity (rsFC) and sex data.
  * 2_SVM_rsfc_sex_prediction.m : Generating predictions using the pre-trained SVM models.
  * 3_SVM_rsfc_sex_correlations.m : Correlating the SVM classification scores (sex predictions) and the sex/gender alignment scores.
  * 4_SVM_rsfc_sex_performance_metrics.m : Peformance metrics calculation.
    
* ## SVR
  
  * 1_SVR_rsfc_sga_females.m : Support Vector Regression. Model training using the rsFC and sex/gender alignment data of females.
  * 2_SVR_rsfc_sga_females_predict.m : Generating predictions using the pre-trained SVR models.
  * 3_SVR_rsfc_sga_females_correlation_r2.m : Correlating the SVR scores (sex/gender alignment predictions) and the orignial sex/gender alignment scores. R-squared calculation.
 
The scripts in the SVM and SVR directories are adaptable, for use with cortical thickness and/or male-specific data.

* ## robustness_check
