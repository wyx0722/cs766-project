%% Basic pipeline

%% Import data
% grab data from open-cv, social force, etc.

%% Setup final data array (training and test)
% Create cell array, data = {}, where data{i} is a 2-d matrix that is 
% F x D, where F is the number of features for this video and D is the
% dimensionality. D should be consistent across videos, but F can change.
% Create a data array for both training and test data.

% V_train = number of training videos
% V_test = number of testing videos
training = cell(V_train);
testing = cell(V_test);

%% Build Codebook

% C = codebook size
% attempts = random restarts for kmeans (try 5)
[codebook] = build_codebook(training, C, attempts);

%% Build model
% this could be one-class SVM, LDA , etc

% LDA example:
% L = number of topics
% codingType = 'llc' or 'vq'
L = 30;
codingType = 'llc';
[alpha, beta] = build_LDA_model(training, codebook, L, codingType)

%% Classify testing
% call the predict function for the model type (need to implement)
