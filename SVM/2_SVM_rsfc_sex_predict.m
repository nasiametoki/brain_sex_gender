rng(1)

%% Extract Predictors from Correlation Matrices and Assign Labels for Aligned and Unaligned Testing Sets
predictors_aligned = zeros(size(subjects_aligned_testing_set,1),77421);
for i = 1:size(subjects_aligned_testing_set)
    predictors_aligned(i,:) = corrmat_vectorize(subjects_aligned_testing_set.Corrmats{i,1})';
end
clear i

predictors_unaligned = zeros(size(subjects_unaligned,1),77421);
for i = 1:size(subjects_unaligned)
    predictors_unaligned(i,:) = corrmat_vectorize(subjects_unaligned.Corrmats{i,1})';
end
clear i

labels_aligned = subjects_aligned_testing_set.sex_recoded;
labels_unaligned = subjects_unaligned.sex_recoded;

%% Regress Out Confounds from Predictors for Aligned and Unaligned Data
[u,~,~] = svd([subjects_aligned_testing_set.pds(:), subjects_aligned_testing_set.race_ethnicity(:), subjects_aligned_testing_set.FD(:)], 'econ');
a_matrix = eye(size(u,1)) - u*u';
predictors_aligned_r = a_matrix*predictors_aligned;

[u,~,~] = svd([subjects_unaligned.pds(:), subjects_unaligned.race_ethnicity(:), subjects_unaligned.FD(:)], 'econ');
a_matrix = eye(size(u,1)) - u*u';
predictors_unaligned_r = a_matrix*predictors_unaligned;

%% Normalize Predictors and Make Predictions with SVM Models
predictors_aligned_norm_allmodels = struct('predictors_aligned', {});
predictions_aligned = struct('predictedValues', {});
scores_aligned_all = struct('scores', {});
predictions_unaligned = struct('predictedValues', {});
scores_unaligned_all = struct('scores', {});

for i = 1:outerK
    predictors_aligned_norm = normalize(predictors_aligned_r,'center', center_values(i).values, 'scale', scale_values(i).values);
    predictors_unaligned_norm = normalize(predictors_unaligned_r,'center', center_values(i).values, 'scale', scale_values(i).values);

    predictors_aligned_norm_allmodels(i).predictors_aligned = predictors_aligned_norm;

    [predictionslabels_aligned, scores_aligned]  = predict(svmModels_aligned(i).models, predictors_aligned_norm);
    predictions_aligned(i).predictedValues = predictionslabels_aligned;
    scores_aligned_all(i).scores = scores_aligned;

    [predictionslabels_unaligned, scores_unaligned]  = predict(svmModels_aligned(i).models, predictors_unaligned_norm);
    predictions_unaligned(i).predictedValues = predictionslabels_unaligned;
    scores_unaligned_all(i).scores = scores_unaligned;

end
