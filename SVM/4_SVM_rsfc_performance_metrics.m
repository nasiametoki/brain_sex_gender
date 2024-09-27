rng(1)

%% Majority Voting for Aligned Predicted Labels Across Cross-Validations
row_predictions = [];
predicted_labels_aligned =[];

for i = 1:size(predictions_aligned(1).predictedValues,1)
    row_predictions = [predictions_aligned(1).predictedValues(i);
        predictions_aligned(2).predictedValues(i);
        predictions_aligned(3).predictedValues(i);
        predictions_aligned(4).predictedValues(i);
        predictions_aligned(5).predictedValues(i)];
    predicted_labels_aligned(i) = mode(row_predictions)';
end

labels_aligned = subjects_aligned_testing_set.sex_recoded;

%% Confusion Matrix for Aligned Data
confusionMat_aligned = confusionmat(labels_aligned, predicted_labels_aligned);

%% Common Performance Metrics for Aligned Classification Model
% accuracy, sensitivity, specificity, the area under the receiver operating characteristic curve (AUC)
% and the Matthews correlation coefficient (MCC)
stats_aligned = confusionmatStats(labels_aligned, predicted_labels_aligned);
mcc_aligned = matthewscorr(confusionMat_aligned);
[~,~,~,AUC_aligned] = perfcurve(labels_aligned, scores_aligned, 1);
rocObj_aligned = rocmetrics(labels_aligned, scores_aligned, 1);

%% Majority Voting for Unaligned Predicted Labels Across Cross-Validations
row_predictions = [];
predicted_labels_unaligned =[];

for i = 1:size(predictions_unaligned(1).predictedValues,1)
    row_predictions = [predictions_unaligned(1).predictedValues(i);
        predictions_unaligned(2).predictedValues(i);
        predictions_unaligned(3).predictedValues(i);
        predictions_unaligned(4).predictedValues(i);
        predictions_unaligned(5).predictedValues(i)];
    predicted_labels_unaligned(i) = mode(row_predictions)';
end

labels_unaligned = subjects_unaligned.sex_recoded;

%% Confusion Matrix for Unaligned Data
confusionMat_unaligned = confusionmat(labels_unaligned, predicted_labels_unaligned);

%% Common Performance Metrics for Unaligned Classification Model
% accuracy, sensitivity, specificity, the area under the receiver operating characteristic curve (AUC)
% and the Matthews correlation coefficient (MCC)
stats_unaligned = confusionmatStats(labels_unaligned, predicted_labels_unaligned);
mcc_unaligned = matthewscorr(confusionMat_unaligned);
[~,~,~,AUC_unaligned] = perfcurve(labels_unaligned, scores_unaligned, 1);
rocObj_unaligned = rocmetrics(labels_unaligned, scores_unaligned, 1);