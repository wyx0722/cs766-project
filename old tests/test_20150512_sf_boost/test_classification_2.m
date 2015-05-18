%% load data
load('labels');
load('labels_train');
labels = double(labels);
labels_train = double(labels_train);

labels(labels == 1) = -1;
labels(labels == 0) = 1;
labels_train(:) = 1;

load('features_train');
load('features');

%% train one-class SVM
num_hypotheses = 10;
num_examples_train = size(features_train, 1);
example_weights = ones(num_examples_train, 1) / num_examples_train;
hypotheses = cell(num_hypotheses, 1);
hypotheses_weights = zeros(num_hypotheses, 1);
for i = 1 : num_hypotheses
    hypotheses{i} = svmtrain(example_weights, labels_train, features_train, '-s 2 -t 0 -n 0.2');
    [labels_train_predict, ~, ~] = svmpredict(labels_train, features_train, hypotheses{i});
    wrong_ids = labels_train ~= labels_train_predict;
    error = sum(wrong_ids .* example_weights);
    if error == 0
        error = min(example_weights) / 10;
    end
    correct_ids = not(wrong_ids);
    example_weights(correct_ids) = example_weights(correct_ids) * (error / (1 - error));
    example_weights = example_weights ./ sum(example_weights);
    hypotheses_weights(i) = log((1 - error) / error);
end

num_examples = size(features, 1);
dec_values_all = zeros(num_examples, 1);
for i = 1 : num_hypotheses
    [~, ~, dec_values] = svmpredict(labels, features, hypotheses{i});
    dec_values_all = dec_values_all + dec_values * hypotheses_weights(i);
end
norm_dec_values = 1 - mat2gray(dec_values_all);

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