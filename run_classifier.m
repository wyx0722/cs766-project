%% General classifier wrapper
%
% Input: 
%    training_hists: 
%    training_labels: 
%    testing_hists: 
%    testing_labels: 
%    classifier_type: 
%
% Output: 
%    predicted_labels: 
%
function [accuracy, ROC_area, predicted_labels, est_values] = run_classifier(training_hists, training_labels, testing_hists, testing_labels, classifier_type)
    % add new classifiers using 'elseif strcmp(classifier_type, <NEW_TYPE>)'

    if strcmp(classifier_type,'1classSVMOnly_Linear')
        addpath('../libsvm-3.20/matlab/');
        model = svmtrain(double(training_labels), training_hists, '-s 2 -t 0');
        [predicted_labels, acc, est_values] = svmpredict(double(testing_labels), testing_hists, model);
        norm_dec_values = mat2gray(est_values);
        accuracy = acc(1);
        rmpath('../libsvm-3.20/matlab/');
    elseif strcmp(classifier_type,'1classSVMOnly_Gauss')
        addpath('../libsvm-3.20/matlab/');
        model = svmtrain(double(training_labels), training_hists, '-s 2 -t 2');
        [predicted_labels, acc, est_values] = svmpredict(double(testing_labels), testing_hists, model);
        norm_dec_values = mat2gray(est_values);
        accuracy = acc(1);
        rmpath('../libsvm-3.20/matlab/');
        
    %elseif strcmp(classifier_type,'ldaDimReduction')
    %    L = 30;
    %    predicted_labels = build_LDA_model(training_hists, testing_hists, L);
    else
        error('Invalid classifier type!')
    end
    
    %% plot ROC curve
    roc_labels = (testing_labels==1)';
    roc_predict_labels = norm_dec_values';
    plotroc(roc_labels, roc_predict_labels);
    [tpr,fpr,~] = roc(roc_labels, roc_predict_labels);
    ROC_area = 0;
    for i = 1 : length(fpr) - 1
        ROC_area = ROC_area + (fpr(i + 1) - fpr(i)) * tpr(i);
    end
end