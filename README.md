# brain_sex_gender

This is the analysis workflow for: ["Brain functional connectivity, but not neuroanatomy, captures the interrelationship between sex and gender in preadolescents"](https://doi.org/10.1101/2024.10.31.621379)

## Overview
* ## SVM

  The SVM (Support Vector Machine) folder contains scripts for SVM classification, prediction, correlation, and peformance metrics calculation
  
  * 1_SVM_rsfc_sex.m : Support Vector Machine classification. Model training using the resting-state functional connectivity (rsFC) and sex data.
  * 2_SVM_rsfc_sex_prediction.m : Generating predictions using the pre-trained SVM models.
  * 3_SVM_rsfc_sex_correlations.m : Correlating the SVM classification scores (sex predictions) and the sex/gender alignment scores.
  * 4_SVM_rsfc_sex_performance_metrics.m : Peformance metrics calculation.
    
* ## SVR

  The SVR (Support Vector Regression) folder contains scripts for SVR analysis, prediction, correlation, and peformance metrics calculation
  
  * 1_SVR_rsfc_sga_females.m : Support Vector Regression. Model training using the rsFC and sex/gender alignment data of females.
  * 2_SVR_rsfc_sga_females_predict.m : Generating predictions using the pre-trained SVR models.
  * 3_SVR_rsfc_sga_females_correlation_r2.m : Correlating the SVR scores (sex/gender alignment predictions) and the orignial sex/gender alignment scores. R-squared calculation.
 
  The scripts in the SVM and SVR directories are adaptable, for use with cortical thickness and/or male-specific data.

* ## robustness_check

  The robustness_check folder contains scripts which perform linear ridge regression using rsFC and sex/gender alignment scores. These scripts have been recreated based on the [original versions](https://zenodo.org/records/10779164) developed by Dr. Elvisha Dhamala, as referenced in this [publication](https://www.science.org/doi/epdf/10.1126/sciadv.adn4202).

  * lrr_females_rsfc.ipynb : Linear ridge regression analysis using rsFC and youth sex/gender alignment scores for females.
  * lrr_females_rsfc_parent.ipynb : Linear ridge regression analysis using rsFC and parent sex/gender alignment scores for females.
  * lrr_males_rsfc.ipynb : Linear ridge regression analysis using rsFC and youth sex/gender alignment scores for males.
  * lrr_males_rsfc_parent.ipynb : Linear ridge regression analysis using rsFC and parent sex/gender alignment scores for males.
