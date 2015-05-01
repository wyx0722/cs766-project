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

%% Build Codebook (done!)
% C = codebook size
% attempts = random restarts for kmeans (try 5)
codebook = build_codebook(training, C, attempts);

%% Extract words from training and test sets (done!)
% build the final classification matrices
valid_coding_types = {'llc', 'vq'};
coding_type = valid_coding_types{1};
training_hists = build_clip_histogram(training, codebook, coding_type);
testing_hists = build_clip_histogram(testing, codebook, coding_type);

%% Build model and classify (basic wrapper function done... can add more types!)
% this could be one-class SVM, LDA , etc
valid_classifier_types = {'lda'}; % current types are 'lda' (add more to wrapper function)
classifier_type = valid_classifier_types(1);
predicted_labels = run_classifier(training_hists, testing_hists, classifier_type);

%% Calculate final outputs
% using predicted_labels, build whatever results we need
