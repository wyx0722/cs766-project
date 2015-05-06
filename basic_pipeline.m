clear
%% Basic pipeline
experiment_directoy = 'test_20150505_2_sf_only/';
% Create the folder if it doesn't exist already.
if ~exist(experiment_directoy, 'dir')
	mkdir(experiment_directoy);
end


%% Import data
% grab data from open-cv, social force, etc.

% ASSUMES THE EXISTENCE OF all_features.mat
load('all_features');

%% Setup final data array (training and test)
% Create cell array, data = {}, where data{i} is a 2-d matrix that is 
% F x D, where F is the number of features for this video and D is the
% dimensionality. D should be consistent across videos, but F can change.
% Create a data array for both training and test data.

% 1 -> use this feature group, 0 -> do not use this feature group
use_DenseTraj = 0;
use_HOF = 0;
use_HOG = 0;
use_MBH = 0;
use_SocialForce = 1;

if exist([experiment_directoy 'data.mat'], 'file') == 2
    load([experiment_directoy 'data.mat']);
    load([experiment_directoy 'labels.mat']);
else
    data = {};
    labels = cell2mat(clip_raw_features.labels)';
    for video = 1:length(clip_raw_features.rawFileNames)
        tmp_mat = [];

        if use_DenseTraj
            tmp_mat = [tmp_mat clip_raw_features.denseTrajectories{video}];
        end
        if use_HOF
            tmp_mat = [tmp_mat clip_raw_features.HOF{video}];
        end
        if use_HOG
            tmp_mat = [tmp_mat clip_raw_features.HOG{video}];
        end
        if use_MBH
            tmp_mat = [tmp_mat clip_raw_features.MBH{video}];
        end
        if use_SocialForce
            tmp_mat = [tmp_mat clip_raw_features.SocialForce{video}];
        end

        data{video} = tmp_mat;
        clear tmp_mat
    end
    save([experiment_directoy 'data.mat'],'data');
    save([experiment_directoy 'labels.mat'],'labels');
end
clear use_DenseTraj use_HOF use_HOG use_MBH use_SocialForce video

% V_train = number of training videos
% V_test = number of testing videos

% pick 5 random videos for training
numVideos = length(clip_raw_features.clipIndexInfo);
indices = [];
for i = 1:numVideos
    indices = [indices; (i * ones(length(clip_raw_features.clipIndexInfo{i}),1))];
end
clear clip_raw_features i

if exist([experiment_directoy 'train_mask.mat'], 'file') == 2
    load([experiment_directoy 'train_mask.mat']);
    load([experiment_directoy 'testing.mat']);
    load([experiment_directoy 'training.mat']);
    load([experiment_directoy 'labels_training.mat']);
else
    random_videos = randperm(numVideos,5);
    train_mask = (labels==1) & ( (indices==random_videos(1)) | (indices==random_videos(2))...
         | (indices==random_videos(3)) | (indices==random_videos(4))  | (indices==random_videos(5))  );
    training = data(train_mask);
    labels_training = labels(train_mask);
    testing = data;
    save([experiment_directoy 'train_mask.mat'],'train_mask');
    save([experiment_directoy 'testing.mat'],'testing');
    save([experiment_directoy 'training.mat'],'training');
    save([experiment_directoy 'labels_training.mat'],'labels_training');
end
clear data random_videos train_mask indices numVideos

%% Build Codebook (done!)
% C = codebook size
% attempts = random restarts for kmeans (try 5)
if exist([experiment_directoy 'codebook.mat'], 'file') == 2
    load([experiment_directoy 'codebook.mat']);
else
    C = 1024;
    attempts = 4;
    codebook = build_codebook(training, C, attempts);
    save([experiment_directoy 'codebook.mat'], 'codebook');
    clear C attempts
end

%% Extract words from training and test sets (done!)
% build the final classification matrices
valid_coding_types = {'llc', 'vq'};

if exist([experiment_directoy 'training_hists.mat'], 'file') == 2
    load([experiment_directoy 'training_hists.mat']);
    load([experiment_directoy 'testing_hists.mat']);
else
    coding_type = valid_coding_types{1};
    training_hists = build_clip_histogram(training, codebook, coding_type);
    testing_hists = build_clip_histogram(testing, codebook, coding_type);
    clear valid_coding_types coding_type testing training
    save([experiment_directoy 'training_hists.mat'],'training_hists');
    save([experiment_directoy 'testing_hists.mat'],'testing_hists');
end

%% Build model and classify (basic wrapper function done... can add more types!)
% this could be one-class SVM, LDA , etc
valid_classifier_types = {'1classSVMOnly_Linear','1classSVMOnly_Poly','1classSVMOnly_Gauss'}; %(add more to wrapper function)
classifier_type = valid_classifier_types(3);
[accuracy, ROC_area, predicted_labels, est_values] = run_classifier(training_hists, labels_training, testing_hists, labels, classifier_type);

