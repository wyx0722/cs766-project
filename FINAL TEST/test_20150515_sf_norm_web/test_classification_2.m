%% training set indices
indices_train = zeros(10, 10);
for i = 1 : 10
    temp = randperm(12);
    indices_train(i, :) = temp(1 : 10);
end
clear i temp;
save 'indices_train' indices_train;

%% load data
load('indices');
load('indices_train');
load('features');
load('labels');
labels = double(labels);
labels(labels == 1) = -1;
labels(labels == 0) = 1;

%% train one-class SVM
for round = 1 : 10;
    mask_train = ismember(indices, indices_train(round, :));
    features_train = features(mask_train, :);
    labels_train = labels(mask_train, :);

    
    model = svmtrain(double(labels_train), features_train, '-s 2 -t 0');
    [predict_labels, ~, dec_values] = svmpredict(double(labels), features, model);
    norm_dec_values = 1 - mat2gray(dec_values);
    
    results(:, round) = norm_dec_values;
end
results = sum(results, 2) / 10;

%% plot ROC curve
roc_labels = (labels==-1)';
roc_predict_labels = results';
plotroc(roc_labels, roc_predict_labels);
[tpr,fpr,thresholds] = roc(roc_labels, roc_predict_labels);
area = 0;
for i = 1 : length(fpr) - 1
    area = area + (fpr(i + 1) - fpr(i)) * tpr(i);
end
area