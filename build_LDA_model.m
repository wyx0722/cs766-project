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

% build data files in correct format for train/test sets (unfortunately nasty)
training_filename = 'lda-c/model/training_data.dat';
testing_filename = 'lda-c/model/testing_data.dat';
word_ind = 1:C;
formatSpec = ' %i:%f';

train_file = fopen(training_filename,'w');
for train_doc = 1:V_train
    non_zero_ind = (train_data(train_doc,:) > 0);
    non_zero_sum = sum(non_zero_ind);
    fprintf(train_file, '%i', non_zero_sum); % 1st number
    fprintf(train_file, formatSpec, [word_ind(non_zero_ind); train_data(train_doc,non_zero_ind)]); % data
    fprintf(train_file, '\n'); % endline
end
fclose(train_file);

test_file = fopen(testing_filename,'w');
for test_doc = 1:V_test
    non_zero_ind = (testing_data(test_doc,:) > 0);
    non_zero_sum = sum(non_zero_ind);
    fprintf(test_file, '%i', non_zero_sum); % 1st number
    fprintf(test_file, formatSpec, [word_ind(non_zero_ind); testing_data(test_doc,non_zero_ind)]); % data
    fprintf(test_file, '\n'); % endline
end
fclose(test_file);

run_string = ['./lda-c/build/lda est 0.75 ' num2str(L) ' lda-c/settings.txt ' training_filename ' random lda-c/model/'];
s = system(run_string);

% Run LDA inference on test data

end