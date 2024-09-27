%% Average Predicted Values from All Cross-Validations
predictions = 0;

for i = 1:numel(predictions)
    predictions = predictions + predictions(i).predictedValues(:,1);
end
predictions = predictions / numel(predictions);

%% Obtain Original Sex/Gender Alignment (SGA) Values
original_sag = females_testing_set.mean_gender_scores;

%% Remove Outliers
outlier = find(isoutlier(predictions));

for i = outlier
    predictions_nooutl = predictions;
    original_sag_nooutl = original_sag;
    predictions_nooutl(i, :) = [];
    original_sag_nooutl(i,:) = [];
end

%% Correlate
[RF_svr, PF_svr] = corrcoef(original_sag_nooutl, predictions_nooutl);

%% R-squared Calculation
SST = sum((original_sag_nooutl - mean(original_sag_nooutl)).^2);
SSE = sum((original_sag_nooutl - predictions_nooutl).^2);
R_squared = 1 - SSE / SST;
