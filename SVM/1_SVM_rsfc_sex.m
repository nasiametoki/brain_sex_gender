rng(1)

%% Extract Predictors from Correlation Matrices and Assign Labels for Training Set
predictors_aligned_training_set = zeros(size(subjects_aligned_training_set,1),77421);
for i = 1:size(subjects_aligned_training_set)
    predictors_aligned_training_set(i,:) = corrmat_vectorize(subjects_aligned_training_set.Corrmats{i,1})';
end
clear i

labels_aligned_training_set = subjects_aligned_training_set.sex_recoded;

%% Set Hyperparameters for Lambda Regularization
% Define the range of lambda values to tune
start_power = -10;
end_power = 15;
step_power = 2;

lambda_values = 2.^(start_power:step_power:end_power);

clear start_power end_power step_power

%% Set Cross-Validation Parameters
% Number of folds for outer and inner cross-validation
outerK = 5;
innerK = 5;

%% Outer Cross-Validation for Generalizability
outerCV = cvpartition(subjects_aligned_training_set.sex_recoded, 'KFold', outerK);
accs = zeros(numel(lambda_values), outerK);

for i = 1:outerK
    predictors_outer_train_aligned = predictors_aligned_training_set(outerCV.training(i), :);
    predictors_outer_test_aligned = predictors_aligned_training_set(outerCV.test(i), :);
    labels_outer_train_aligned = labels_aligned_training_set(outerCV.training(i), :);
    labels_outer_test_aligned = labels_aligned_training_set(outerCV.test(i), :);

    subjects_train_outer_aligned = subjects_aligned_training_set(outerCV.training(i), :);
    subjects_test_outer_aligned = subjects_aligned_training_set(outerCV.test(i), :);

    %% Regress Out Confounds and Normalize Data
    [u,~,~] = svd([subjects_train_outer_aligned.pds(:), subjects_train_outer_aligned.race_ethnicity(:), subjects_train_outer_aligned.FD(:)], 'econ');
    a_matrix = eye(size(u,1)) - u*u';
    predictors_train_aligned_r = a_matrix*predictors_outer_train_aligned;

    [u,~,~] = svd([subjects_test_outer_aligned.pds(:), subjects_test_outer_aligned.race_ethnicity(:), subjects_test_outer_aligned.FD(:)], 'econ');
    a_matrix = eye(size(u,1)) - u*u';
    predictors_test_aligned_r = a_matrix*predictors_outer_test_aligned;

    [predictors_train_outer_aligned_norm, C, S] = normalize(predictors_train_aligned_r);
    predictors_test_aligned_norm = normalize(predictors_test_aligned_r, 'center', C, 'scale', S);
    
    center_values = struct('values', {});
    center_values(i).values = C;
    scale_values = struct('values', {});
    scale_values(i).values = S;


    predictors_test_aligned_norm_allmodels = struct('predictors_test_aligned', {});
    predictors_test_aligned_norm_allmodels(i).predictors_test_aligned = predictors_test_aligned_norm;

    %% Nested Cross-Validation to Tune Lambda
    innerCV = cvpartition(subjects_train_outer_aligned.sex_recoded, 'KFold', innerK);

    for lambda_index = 1:length(lambda_values)
        lambda = lambda_values(lambda_index);

        for m = 1:innerK
            predictors_train_inner = predictors_train_outer_aligned_norm(innerCV.training(m), :);
            labels_train_inner = labels_outer_train_aligned(innerCV.training(m));
            predictors_test_inner = predictors_train_outer_aligned_norm(innerCV.test(m), :);
            labels_test_inner = labels_outer_train_aligned(innerCV.test(m));

            %% Compute Class Weights for Inner-Cross Validation
            unique_scores_inner = unique(labels_train_inner);
            proportions_inner = zeros(length(unique_scores_inner),1);
            weights_inner = zeros(length(unique_scores_inner),1);
            all_weights_inner = zeros(size(labels_train_inner));

            for b = 1:size(unique_scores_inner,1)
                proportions_inner(b,1) = sum(labels_train_inner == unique_scores_inner(b))/size(labels_train_inner,1);
                weights_inner(b,1) = 1/proportions_inner(b,1);
            end

            for c = 1:length(labels_train_inner)
                index = find(unique_scores_inner == labels_train_inner(c));
                all_weights_inner(c) = weights_inner(index);
            end

            %% Train Model and Evaluate on Inner Validation Set
            model = fitclinear(predictors_train_inner, labels_train_inner, 'Lambda', lambda, 'Weights', all_weights_inner);
            predlabels_inner = predict(model, predictors_test_inner);
            confusionmat_inner = confusionmat(labels_test_inner, predlabels_inner);
            acc_inner = zeros(outerK,1);
            acc_inner(m) = (confusionmat_inner(1,1) + confusionmat_inner(2,2)) / sum(confusionmat_inner(:)) * 100;
            accs(lambda_index, i) = mean(acc_inner);

        end
    end

    mean_accs = mean(accs, 2);
    [max_inner_accuracy, max_inner_accuracy_index] = max(mean_accs);
    bestInnerLambda = lambda_values(max_inner_accuracy_index);

    %% Compute Class Weights for Outer Training Set
    males_proportion = sum(labels_outer_train_aligned == -1)/size(labels_outer_train_aligned,1);
    females_proportion = sum(labels_outer_train_aligned == 1)/size(labels_outer_train_aligned,1);

    males_weight = 1 / males_proportion;
    females_weight = 1 / females_proportion;

    classWeights = ones(size(labels_outer_train_aligned, 1), 1);
    classWeights(labels_outer_train_aligned == -1) = males_weight;
    classWeights(labels_outer_train_aligned == 1) = females_weight;

    %% Train Final SVM Models on Outer Folds
    svmModel = fitclinear(predictors_train_outer_aligned_norm, labels_outer_train_aligned, 'Lambda', bestInnerLambda, 'Weights', classWeights);
    svmModels_aligned = zeros(outerK,1);
    svmModels_aligned(i).models = svmModel;

    %% Haufe Transformation for Interpretability
    predictionslabels_outer_aligned  = predict(svmModels_aligned(i).models, predictors_train_outer_aligned_norm);
    predictions_aligned = struct('predictedValues', {});
    predictions_aligned(i).predictedValues = predictionslabels_outer_aligned;
    haufeWeights_aligned = struct('weights', {});
    haufeWeights_aligned(i).weights = CBIG_ICCW_cov_matrix(predictors_train_outer_aligned_norm, predictionslabels_outer_aligned);

end