%% load data
load('labels');
load('labels_train');
labels = double(labels);
labels_train = double(labels_train);

labels(labels == 1) = -1;
labels(labels == 0) = 1;
labels_train(labels_train == 0) = 1;

load('features_train');
load('features');

%% train one-class SVM
model = svmtrain(labels_train, features_train, '-s 2 -t 0');
[predict_labels, ~, dec_values] = svmpredict(labels, features, model);
norm_dec_values = 1 - mat2gray(dec_values);

%% plot ROC curve
roc_labels = (labels==-1)';
roc_predict_labels = norm_dec_values';
plotroc(roc_labels, roc_predict_labels);
[tpr,fpr,thresholds] = roc(roc_labels, roc_predict_labels);
area = 0;
for i = 1 : length(fpr) - 1
    area = area + (fpr(i + 1) - fpr(i)) * tpr(i);
end
area