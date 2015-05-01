%% LDA Classifier
% Trains Latent Dirichlet Allocation model for normal video clips then classifies test data
%
% Input: 
%    train_data: a 2-d matrix of V training clips by C (codebook size) words
%    testing_data: a 2-d matrix of V testing clips by C (codebook size) words
%    L: is the number of topics
%
% Output: 
%    alpha: parameter of Dirichlet distribution
%    beta: parameters for topic distributions
%
function [predicted_labels] = build_LDA_model(train_data, testing_data, L)

[V_train, ~] = size(train_data);
[V_test, C] = size(testing_data);

% Run LDA estimation on training (need to adjust to use lda-c)

% Run LDA inference on test data

end