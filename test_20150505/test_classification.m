clear
%% load data
load('labels');
load('labels_train');

labels = double(labels);
labels_train = double(labels_train);

labels(labels == 1) = -1;
labels(labels == 0) = 1;
labels_train(labels_train == 0) = 1;

LDA_features_train = load('final.gamma');
%normalize
norm_train = max(LDA_features_train);
norm_train = repmat(norm_train, [length(LDA_features_train) 1]);
LDA_features_train = LDA_features_train./norm_train;

LDA_features_test = load('ESTIMATE-gamma.dat');
norm_test = max(LDA_features_test);
norm_test = repmat(norm_test, [length(LDA_features_test) 1]);
LDA_features_test = LDA_features_test./norm_test;

%% train one-class SVM
addpath('../libsvm-3.20/matlab/');
model = svmtrain(double(labels_train), LDA_features_train, '-s 2 -t 2 -n 0.05');
[predict_labels, ~, dec_values] = svmpredict(double(labels), LDA_features_test, model);
%predict_labels = sign(predict_labels);
norm_dec_values = 1 - mat2gray(dec_values);
rmpath('../libsvm-3.20/matlab/');

%% plot ROC curve
roc_labels = (labels==1)';
roc_predict_labels = norm_dec_values';
plotroc(roc_labels, roc_predict_labels);
[tpr,fpr,thresholds] = roc(roc_labels, roc_predict_labels);
area = 0;
for i = 1 : length(fpr) - 1
    area = area + (fpr(i + 1) - fpr(i)) * tpr(i);
end
area