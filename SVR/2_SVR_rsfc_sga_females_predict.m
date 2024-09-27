rng(1)

%% Extract Predictors from Correlation Matrices and Assign Labels for Females Testing Set
predictors = zeros(size(females_testing_set,1),77421);
for i = 1:size(females_testing_set)
    predictors(i,:) = corrmat_vectorize(females_testing_set.Corrmats{i,1})';
end

clear i

labels = females_testing_set.sex_recoded;

%% Normalize Predictors and Make Predictions with SVM Models
predictions = struct('predictedValues', {});
for i = 1:outerK
    predictors_norm = normalize(predictors,'center', center_values(i).values, 'scale', scale_values(i).values);
    predictionslabels  = predict(models_f(i).model, predictors_norm);
    predictions(i).predictedValues = predictionslabels;
end

