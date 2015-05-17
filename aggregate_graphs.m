%% Aggregate graphs for random trials

base_dir = 'FINAL TEST/';
exp_names = {'random_train_1/','random_train_2/','random_train_3/','random_train_4/','random_train_5/'};
%exp_names = {'random_train_1/','random_train_2/','random_train_4/','random_train_5/'};
feat_names = {'Dense Trajectories', 'Social Force (Mike)', 'Social Force (Ke)', 'HOF', 'HOG', 'MBH'};
num_exp = length(exp_names);

%load labels
load('all_features.mat');
labels = cell2mat(clip_raw_features.labels)';
agg_labels = repmat(labels,num_exp,1);
ROC_agg_labels = (agg_labels == -1)';
clear clip_raw_features labels agg_labels

% aggregate results vars

% ROC area
DT_avg_ROC_area = 0;
SFM_avg_ROC_area = 0;
SFK_avg_ROC_area = 0;
HOF_avg_ROC_area = 0;
HOG_avg_ROC_area = 0;
MBH_avg_ROC_area = 0;

% ROC curves
DT_est_val = [];
SFM_est_val = [];
SFK_est_val = [];
HOF_est_val = [];
HOG_est_val = [];
MBH_est_val = [];

for this_exp = 1:num_exp
    exp_base_dir = [base_dir exp_names{this_exp}];
    load([exp_base_dir 'results.mat']);
    
    % ROC area
    DT_avg_ROC_area = DT_avg_ROC_area + results.roc_areas(1);
    SFM_avg_ROC_area = SFM_avg_ROC_area + results.roc_areas(2);
    SFK_avg_ROC_area = SFK_avg_ROC_area + results.roc_areas(3);
    HOF_avg_ROC_area = HOF_avg_ROC_area + results.roc_areas(4);
    HOG_avg_ROC_area = HOG_avg_ROC_area + results.roc_areas(5);
    MBH_avg_ROC_area = MBH_avg_ROC_area + results.roc_areas(6);
    
    % ROC curve
    DT_est_val = [DT_est_val; results.est(:,1)];
    SFM_est_val = [SFM_est_val; results.est(:,2)];
    SFK_est_val = [SFK_est_val; results.est(:,3)];
    HOF_est_val = [HOF_est_val; results.est(:,4)];
    HOG_est_val = [HOG_est_val; results.est(:,5)];
    MBH_est_val = [MBH_est_val; results.est(:,6)];
    clear results exp_base_dir
end
clear this_exp exp_names

%% finalize

% ROC area
DT_avg_ROC_area = DT_avg_ROC_area ./ num_exp;
SFM_avg_ROC_area = SFM_avg_ROC_area ./ num_exp;
SFK_avg_ROC_area = SFK_avg_ROC_area ./ num_exp;
HOF_avg_ROC_area = HOF_avg_ROC_area ./ num_exp;
HOG_avg_ROC_area = HOG_avg_ROC_area ./ num_exp;
MBH_avg_ROC_area = MBH_avg_ROC_area ./ num_exp;
ROC_areas = [ DT_avg_ROC_area, SFM_avg_ROC_area, SFK_avg_ROC_area, HOF_avg_ROC_area, HOG_avg_ROC_area, MBH_avg_ROC_area ];

b = figure('Position',[100,100,800,600]);
%bar(ROC_areas);

hold on

bar(0,ROC_areas(1),'b');
text(0-0.25,ROC_areas(1)+0.013,num2str(ROC_areas(1)));

bar(1,ROC_areas(2),'r');
text(1-0.25,ROC_areas(2)+0.013,num2str(ROC_areas(2)));

bar(2,ROC_areas(3),'g');
text(2-0.25,ROC_areas(3)+0.013,num2str(ROC_areas(3)));

bar(3,ROC_areas(4),'c');
text(3-0.25,ROC_areas(4)+0.013,num2str(ROC_areas(4)));

bar(4,ROC_areas(5),'m');
text(4-0.25,ROC_areas(5)+0.013,num2str(ROC_areas(5)));

bar(5,ROC_areas(6),'k');
text(5-0.25,ROC_areas(6)+0.013,num2str(ROC_areas(6)));

hold off
legend(feat_names);

%set(gca,'xticklabel', feat_names);
set(gca,'xticklabel', {'','','','','',''});

print(b,[base_dir 'average_ROC_area.png'], '-dpng');

%% ROC curve
DT_est_val = 1 - mat2gray(DT_est_val)';
SFM_est_val = 1 - mat2gray(SFM_est_val)';
SFK_est_val = 1 - mat2gray(SFK_est_val)';
HOF_est_val = 1 - mat2gray(HOF_est_val)';
HOG_est_val = 1 - mat2gray(HOG_est_val)';
MBH_est_val = 1 - mat2gray(MBH_est_val)';

r = figure();
plotroc(ROC_agg_labels, DT_est_val, feat_names{1}, ...
    ROC_agg_labels, SFM_est_val, feat_names{2}, ... 
    ROC_agg_labels, SFK_est_val, feat_names{3}, ... 
    ROC_agg_labels, HOF_est_val, feat_names{4}, ... 
    ROC_agg_labels, HOG_est_val, feat_names{5}, ... 
    ROC_agg_labels, MBH_est_val, feat_names{6} );
set(r,'Position',[100,100,800,600]);
ylabel('True Abnormal Rate');
xlabel('False Abnormal Rate');

print(r,[base_dir 'aggregate_ROC_curves.png'], '-dpng');

close(b)
close(r)
clear base_dir feat_names num_exp r b