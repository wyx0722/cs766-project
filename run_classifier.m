%% General classifier wrapper
%
% Input: 
%    training_hists: 
%    testing_hists: 
%    classifier_type: 
%
% Output: 
%    predicted_labels: 
%
function [predicted_labels] = run_classifier(training_hists, testing_hists, classifier_type)
    % add new classifiers using 'elseif strcmp(classifier_type, <NEW_TYPE>)'

    if strcmp(classifier_type,'lda')
        L = 30;
        predicted_labels = build_LDA_model(training_hists, testing_hists, L);
    else
        error('Invalid classifier type!')
    end
end