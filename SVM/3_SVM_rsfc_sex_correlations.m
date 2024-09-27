rng(1)

%% Average Aligned and Unaligned Predicted Scores Across Cross-Validations
scores_aligned = zeros(size(scores_aligned_all(1).scores, 1), 1);
for i = 1:numel(scores_aligned_all)
    scores_aligned = scores_aligned + scores_aligned_all(i).scores(:, 2);
end
scores_aligned = scores_aligned / numel(scores_aligned_all);

scores_unaligned = zeros(size(scores_unaligned_all(1).scores, 1), 1);
for i = 1:numel(scores_unaligned_all)
    scores_unaligned = scores_unaligned + scores_unaligned_all(i).scores(:, 2);
end
scores_unaligned = scores_unaligned / numel(scores_unaligned_all);

%% Obtain Original Sex/Gender Alignment (SGA) Values for Both Aligned and Unaligned Groups
original_sga_aligned = subjects_aligned_testing_set.mean_gender_scores_recoded;
original_sga_unaligned = subjects_unaligned.mean_gender_scores_recoded;

%% Combine SVM and SGA Scores, and Remove Outliers
svm_scores = [scores_unaligned; scores_aligned];
sga_scores = [original_sga_unaligned; original_sga_aligned];

outlier = find(isoutlier(svm_scores));

for i = outlier
    svm_scores_nooutl = svm_scores;
    sga_scores_nooutl = sga_scores;
    svm_scores_nooutl(i, :) = [];
    sga_scores_nooutl(i, :) = [];
end

%% Split Scores Based on Sex and Correlate for Females and Males
idx_sga_scores_females = (sga_scores_nooutl > 0);
idx_sga_scores_males = (sga_scores_nooutl < 0);
svm_scores_females = svm_scores_nooutl(idx_sga_scores_females);
svm_scores_males = svm_scores_nooutl(idx_sga_scores_males);
sga_scores_females = sga_scores_nooutl(idx_sga_scores_females);
sga_scores_males = sga_scores_nooutl(idx_sga_scores_males);

[RF,PF] = corrcoef(svm_scores_females, sga_scores_females);
[RM,PM] = corrcoef(svm_scores_males*(-1), sga_scores_males*(-1));

%% Exclude Aligned Group and Correlate for Females and Males
idx_sga_scores_females_unaligned = (sga_scores_nooutl > 0 & sga_scores_nooutl < 5);
idx_sga_scores_males_unaligned = (sga_scores_nooutl < 0 & sga_scores_nooutl > -5);
svm_scores_females_unaligned = svm_scores_nooutl(idx_sga_scores_females_unaligned);
svm_scores_males_unaligned = svm_scores_nooutl(idx_sga_scores_males_unaligned);
sga_scores_females_unaligned = sga_scores_nooutl(idx_sga_scores_females_unaligned);
sga_scores_males_unaligned = sga_scores_nooutl(idx_sga_scores_males_unaligned);

[RF_unaligned, PF_unaligned] = corrcoef(svm_scores_females_unaligned, sga_scores_females_unaligned);
[RM_unaligned, PM_unaligned] = corrcoef(svm_scores_males_unaligned*(-1), sga_scores_males_unaligned*(-1));

%%
% sga_scores_males*(-1) -> to get sga scores back to the original scale (1-5)
% svm_scores_males*(-1) -> We are using the second column from MATLAB "predict" hence a positive score indicates
% that x is predicted to be a female. A negative score indicates male. We multiply by -1
% so that higher raw scores suggest a stronger correspondence to the neural connectivity
% patterns associated with the sex commonly labeled as either female or male 
