%% load data
load('labels');
load('labels_train');
load('features');
load('features_train');

%% train one-class SVM
model = svmtrain(double(labels_train), features_train, '-s 2 -t 0');
[~, ~, dec_values] = svmpredict(double(labels), features, model);
norm_dec_values = 1 - mat2gray(dec_values);

%% plot ROC curve
plotroc(labels', norm_dec_values');
[tpr,fpr,thresholds] = roc(labels', norm_dec_values');
area = 0;
for i = 1 : length(fpr) - 1
    area = area + (fpr(i + 1) - fpr(i)) * tpr(i);
end
area