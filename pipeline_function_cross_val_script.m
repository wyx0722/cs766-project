%% basic test for 5 random training sets

%% setup feature params
% 1 -> use this feature group, 0 -> do not use this feature group
params.use_DenseTraj = 1;
params.use_HOF = 0;
params.use_HOG = 0;
params.use_MBH = 0;
params.use_SocialForce_Mike = 0;
params.use_SocialForce_Ke = 0;
% params.use_DesiredVelocity = 0;
% params.use_DesiredVelocityMag = 0;
% params.use_ActualVelocity = 0;
% params.use_ActualVelocityMag = 0;

% Codebook parameters
% C = codebook size
% attempts = random restarts for kmeans (try 5)
params.C = 1024;
params.attempts = 1;

% Coding parameters
% 'llc' or 'vq'
params.coding_type = 'vq';

% Classifier options
% '1classSVMOnly_Linear', '1classSVMOnly_Poly', or '1classSVMOnly_Gauss'
params.classifier_type = '1classSVMOnly_Linear';

%% other params

ALL_PARAMS = {};
ALL_PARAMS{1} = params;

p2 = params;
p2.use_DenseTraj = 0;
p2.use_SocialForce_Mike = 1;
ALL_PARAMS{2} = p2;

p3 = params;
p3.use_DenseTraj = 0;
p3.use_SocialForce_Ke = 1;
ALL_PARAMS{3} = p3;

p4 = params;
p4.use_DenseTraj = 0;
p4.use_HOF = 1;
ALL_PARAMS{4} = p4;

p5 = params;
p5.use_DenseTraj = 0;
p5.use_HOG = 1;
ALL_PARAMS{5} = p5;

p6 = params;
p6.use_DenseTraj = 0;
p6.use_MBH = 1;
ALL_PARAMS{6} = p6;

%% test all params
tests = {'DT','SF_M','SF_K','HOF','HOG','MBH'}; % names for tests using same training set
base_dir = 'FINAL TEST/';
%exp_names = {'random_train_1/','random_train_2/','random_train_3/','random_train_4/','random_train_5/'}; % change when training mask changes
exp_names = {'random_train_2/'}; % change when training mask changes

% repeat test 5 times with random training sets
for this_exp = 1:length(exp_names)
    exp_base_dir = [base_dir exp_names{this_exp}];
    datafile = 'all_features';
    accuracies = [];
    predicted_labels_ALL = [];
    est_values_ALL = [];
    ROC_areas = [];
    for this_test = 1:length(ALL_PARAMS)
        disp(['START TEST ' num2str(this_test) '...'])
        [accuracy, ROC_area, predicted_labels, est_values, full_string] = basic_pipeline_function(datafile, tests{this_test}, exp_base_dir, ALL_PARAMS{this_test});
        accuracies = [accuracies; accuracy];
        predicted_labels_ALL = [predicted_labels_ALL, predicted_labels];
        est_values_ALL = [est_values_ALL, est_values];
        ROC_areas = [ROC_areas; ROC_area];
        print(gcf,[exp_base_dir 'ROC_' tests{this_test}],'-dpng')
        close(gcf);
        disp(['...END TEST ' num2str(this_test)])
    end
    results.tests = tests;
    results.acc = accuracies;
    results.roc_areas = ROC_areas;
    results.pred = predicted_labels_ALL;
    results.est = est_values_ALL;
    save([exp_base_dir 'results.mat'],'results');
end