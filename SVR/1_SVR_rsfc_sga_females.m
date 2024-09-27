rng(1)

%% Extract Predictors from Correlation Matrices and Assign Labels for Training Set
predictors_females = zeros(size(females_training_set,1),77421);
for i = 1:size(females_training_set)
    predictors_females(i,:) = (corrmat_vectorize(females_training_set.Corrmats{i,1})');
end
clear i

labels_females = females_training_set.mean_gender_scores;

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

%% Initialize variables to store results
cvMSE = zeros(length(lambda_values), 1);
models_f = struct('model', {});

%% Outer Cross-Validation for Generalizability
outerCV = cvpartition(size(females_training_set,1), 'KFold', outerK);

for i = 1:outerK
    predictors_females_train = predictors_females(outerCV.training(i), :);
    labels_females_train = labels_females(outerCV.training(i));
    predictors_females_test = predictors_females(outerCV.test(i), :);
    labels_females_test = labels_females(outerCV.test(i));

    train = females_training_set(outerCV.training(i), :);
    test = females_training_set(outerCV.test(i), :);

    %% Normalize Data
    [predictors_females_train_norm, C, S] = normalize(predictors_females_train);
    predictors_females_test_norm = normalize(predictors_females_test, 'center', C, 'scale', S);

    center_values = struct('values', {});
    center_values(i).values = C;
    scale_values = struct('values', {});
    scale_values(i).values = S;

    %% Nested Cross-Validation to Tune Lambda
    for j = 1:length(lambda_values)

        params_inner = struct('Lambda', lambda_values(j));

        innerCV = cvpartition(size(train,1), 'KFold', innerK);
        innerMSE = zeros(innerK, 1);

        for m = 1:innerK
            predictors_females_train_inner = predictors_females_train_norm(innerCV.training(m), :);
            labels_females_train_inner = labels_females_train(innerCV.training(m));
            predictors_females_test_inner = predictors_females_train_norm(innerCV.test(m), :);
            labels_females_test_inner = labels_females_train(innerCV.test(m));

            %% Compute Weights for Inner-Cross Validation
            unique_scores_inner = unique(labels_females_train_inner);
            proportions_inner = zeros(length(unique_scores_inner),1);
            weights_inner = zeros(size(unique_scores_inner));
            all_weights_inner = zeros(size(labels_females_train_inner));

            for b = 1:size(unique_scores_inner,1)
                proportions_inner(b,1) = sum(labels_females_train_inner == unique_scores_inner(b))/size(labels_females_train_inner,1);
                weights_inner(b,1) = 1/proportions_inner(b,1);
            end

            for c = 1:length(labels_females_train_inner)
                index = find(unique_scores_inner == labels_females_train_inner(c));
                all_weights_inner(c) = weights_inner(index);
            end

            %% Train Model and Evaluate on Inner Validation Set
            model = fitrlinear(predictors_females_train_inner', labels_females_train_inner, 'ObservationsIn','columns', ...
                'Learner','svm','Solver','dual','Regularization','ridge', 'Lambda', params_inner.Lambda, 'Weights', all_weights_inner);
            predY_inner = predict(model, predictors_females_test_inner);

            % Compute Mean Squared Error (MSE)
            innerMSE(m) = mean((labels_females_test_inner - predY_inner).^2);
        end

        % Compute Average Inner Cross-Validation MSE
        cvMSE(j,1) = mean(innerMSE);

    end

    % Find Hyperparameters with Minimum Cross-Validation Error
    [minCV, idx] = min(cvMSE(:));
    minLambda = lambda_values(idx);
    params = struct('Lambda', minLambda);

    %% Compute Weights for Outer Training Set
    unique_scores = unique(labels_females_train);
    proportions = zeros(lenght(unique_scores),1);
    weights = zeros(length(unique_scores),1);
    all_weights = zeros(size(labels_females_train));

    for l = 1:size(unique_scores,1)
        proportions(l,1) = sum(labels_females_train == unique_scores(l))/size(labels_females_train,1);
        weights(l,1) = 1/proportions(l,1);
    end

    for n = 1:length(labels_females_train)
        index = find(unique_scores == labels_females_train(n));
        all_weights(n) = weights(index);
    end

    %% Train Final SVM Models on Outer Folds
    [finalModel_f, FitInfo_f] = fitrlinear(predictors_females_train_norm', labels_females_train, 'ObservationsIn','columns', ...
        'Learner','svm','Solver','dual','Regularization','ridge', 'Lambda', params.Lambda, 'Weights', all_weights);

    models_f(i).model = finalModel_f;
end
