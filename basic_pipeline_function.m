%% Basic pipeline encapsulated as a function
% Inputs:
%    datafile: matrix of all data features [should be 'all_features']
%    experiment_name: name for experiment specific output directory
%    experiment_directory_base: name for base output directory
%    params: options to use (see comments in 'Check parameters')
function [accuracy, ROC_area, predicted_labels, est_values, full_string] = basic_pipeline_function(datafile, experiment_name, experiment_directory_base, params)

common_directory = [experiment_directory_base 'common/'];
experiment_directory = [experiment_directory_base experiment_name '/'];

% Create the folder if it doesn't exist already.
if ~exist(experiment_directory_base, 'dir')
	mkdir(experiment_directory_base);
end
if ~exist(experiment_directory, 'dir')
	mkdir(experiment_directory);
end
if ~exist(common_directory, 'dir')
	mkdir(common_directory);
end


%% Import data
% grab data from open-cv, social force, etc.

% ASSUMES THE EXISTENCE OF all_features.mat
disp('...loading data...');
load(datafile);

%% Check parameters
if(~exist('params','var'))
    % feature params
    % 1 -> use this feature group, 0 -> do not use this feature group
    params.use_DenseTraj = 0;
    params.use_HOF = 0;
    params.use_HOG = 0;
    params.use_MBH = 0;
    params.use_SocialForce_Mike= 0;
    params.use_SocialForce_Ke = 1;
    %params.use_DesiredVelocity = 0;
    %params.use_DesiredVelocityMag = 0;
    %params.use_ActualVelocity = 0;
    %params.use_ActualVelocityMag = 0;
    
    % Codebook parameters
    % C = codebook size
    % attempts = random restarts for kmeans (try 5)
    params.C = 1024;
    params.attempts = 1;
    
    % Coding parameters
    % 'llc' or 'vq'
    params.coding_type = 'vq';
    
    % Classifier options
    % '1classSVMOnly_Linear', '1classSVMOnly_Poly', or '1classSVMOnly_Gauss'
    params.classifier_type = '1classSVMOnly_Linear';
end
if(~isfield(params,'use_DenseTraj'))
    params.use_DenseTraj = 0;
end
if(~isfield(params,'use_HOF'))
    params.use_HOF = 0;
end
if(~isfield(params,'use_HOG'))
    params.use_HOG = 0;
end
if(~isfield(params,'use_MBH'))
    params.use_MBH = 0;
end
if(~isfield(params,'use_SocialForce_Mike'))
    params.use_SocialForce_Mike = 0;
end
if(~isfield(params,'use_SocialForce_Ke'))
    params.use_SocialForce_Ke = 1;
end
% if(~isfield(params,'use_DesiredVelocity'))
%     params.use_DesiredVelocity = 0;
% end
% if(~isfield(params,'use_DesiredVelocityMag'))
%     params.use_DesiredVelocityMag = 0;
% end
% if(~isfield(params,'use_ActualVelocity'))
%     params.use_ActualVelocity = 0;
% end
% if(~isfield(params,'use_ActualVelocityMag'))
%     params.use_ActualVelocityMag = 0;
% end
if(~isfield(params,'C'))
    params.C = 1024;
end
if(~isfield(params,'attempts'))
    params.attempts = 1;
end
if(~isfield(params,'coding_type'))
    params.coding_type = 'vq';
end
if(~isfield(params,'classifier_type'))
    params.classifier_type = '1classSVMOnly_Linear';
end

%% build iterator from options for loops
feature_to_use = {};
full_string = '';
if params.use_DenseTraj
    feature_to_use{end+1} = 'DT';
    full_string = [full_string '_' feature_to_use{end}];
end
if params.use_HOF
    feature_to_use{end+1} = 'HOF';
    full_string = [full_string '_' feature_to_use{end}];
end
if params.use_HOG
    feature_to_use{end+1} = 'HOG';
    full_string = [full_string '_' feature_to_use{end}];
end
if params.use_MBH
    feature_to_use{end+1} = 'MBH';
    full_string = [full_string '_' feature_to_use{end}];
end
if params.use_SocialForce_Mike
    feature_to_use{end+1} = 'SFM';
    full_string = [full_string '_' feature_to_use{end}];
end
if params.use_SocialForce_Ke
    feature_to_use{end+1} = 'SFK';
    full_string = [full_string '_' feature_to_use{end}];
end
% if params.use_DesiredVelocity
%     feature_to_use{end+1} = 'DVel';
%     full_string = [full_string '_' feature_to_use{end}];
% end
% if params.use_DesiredVelocityMag
%     feature_to_use{end+1} = 'DVelMag';
%     full_string = [full_string '_' feature_to_use{end}];
% end
% if params.use_ActualVelocity
%     feature_to_use{end+1} = 'AVel';
%     full_string = [full_string '_' feature_to_use{end}];
% end
% if params.use_ActualVelocityMag
%     feature_to_use{end+1} = 'AVelMag';
%     full_string = [full_string '_' feature_to_use{end}];
% end

%% Setup training/testing masks
disp('...building training mask...');
labels = cell2mat(clip_raw_features.labels)';

% pick 5 random videos for training
numVideos = length(clip_raw_features.clipIndexInfo);
indices = [];
for i = 1:numVideos
    indices = [indices; (i * ones(length(clip_raw_features.clipIndexInfo{i}),1))];
end

% train mask
if exist([common_directory 'train_mask.mat'], 'file') == 2
    load([common_directory 'train_mask.mat']);
else
	train_mask = [];
	if(isfield(params,'specific_train_videos'))
		videos = params.specific_train_videos;
		train_mask = (labels==1) & ( (indices==videos(1)) | (indices==videos(2))...
         | (indices==videos(3)) | (indices==videos(4))  | (indices==videos(5))  );
    else
        all_ind = 1:length(labels);
        pos_ind = all_ind(labels==1);
        shuffle = pos_ind(randperm(length(pos_ind)));
        shuffle = shuffle(1:round(length(pos_ind)/2)); %50 of positive example for testing
        train_mask = false(size(labels));
        train_mask(shuffle) = true;
	end

    save([common_directory 'train_mask.mat'],'train_mask');
end
clear indices random_videos numVideos

% train labels
labels_training = labels(train_mask);

%% Build codebooks and feature matrices
disp('...building codebook and histogram (this may take time)...');
testing_hists = [];
if exist([experiment_directory 'testing_hists.mat'], 'file') == 2
    load([experiment_directory 'testing_hists.mat']);
else
    for i = 1:length(feature_to_use)
		% hist
		hist_name = ['hist_' feature_to_use{i} '.mat'];
		this_hist = [];
        if exist([common_directory hist_name], 'file') == 2
            load([common_directory hist_name]);
        else
			%data
			this_data = {};
			for video = 1:length(clip_raw_features.rawFileNames)
				tmp_mat = [];
				if strcmp(feature_to_use{i}, 'DT')
					tmp_mat = clip_raw_features.denseTrajectories{video};
				elseif strcmp(feature_to_use{i}, 'HOF')
					tmp_mat = clip_raw_features.HOF{video};
				elseif strcmp(feature_to_use{i}, 'HOG')
					tmp_mat = clip_raw_features.HOG{video};
				elseif strcmp(feature_to_use{i}, 'MBH')
					tmp_mat = clip_raw_features.MBH{video};
				elseif strcmp(feature_to_use{i}, 'SFM')
					tmp_mat = clip_raw_features.SocialForce_Mike{video};
                elseif strcmp(feature_to_use{i}, 'SFK')
					tmp_mat = clip_raw_features.SocialForce_Ke{video};
% 				elseif strcmp(feature_to_use{i}, 'DVel')
% 					tmp_mat = clip_raw_features.DesiredVelocity{video};
% 				elseif strcmp(feature_to_use{i}, 'DVelMag')
% 					tmp_mat = clip_raw_features.DesiredVelocityMag{video};
% 				elseif strcmp(feature_to_use{i}, 'AVel')
% 					tmp_mat = clip_raw_features.ActualVelocity{video};
% 				elseif strcmp(feature_to_use{i}, 'AVelMag')
% 					tmp_mat = clip_raw_features.ActualVelocityMag{video};
				else
					error('PROBLEM WITH feature_to_use VALUE!')
				end
				this_data{video} = tmp_mat;
				clear tmp_mat
			end
		
			%codebook
			codebook_name = ['codebook_' feature_to_use{i} '.mat'];
			this_codebook = [];
			if exist([common_directory codebook_name], 'file') == 2
				load([common_directory codebook_name]);
			else
				this_training = this_data(train_mask);
				this_codebook = build_codebook(this_training, params.C, params.attempts);
				save([common_directory codebook_name], 'this_codebook');
			end
		
			this_hist = build_clip_histogram(this_data, this_codebook, params.coding_type);
			save([common_directory hist_name], 'this_hist');
        end
		
		testing_hists = [testing_hists, this_hist];
		clear codebook_name this_codebook this_training this_data this_hist hist_name
    end
	save([experiment_directory 'testing_hists.mat'],'testing_hists');
end
clear feature_to_use
training_hists = testing_hists(train_mask,:);
clear train_mask

%% Build model and classify (basic wrapper function done... can add more types!)
% this could be one-class SVM, LDA , etc
disp('...classifying...');
[accuracy, ROC_area, predicted_labels, est_values] = run_classifier(training_hists, labels_training, testing_hists, labels, params.classifier_type);
clear training_hists labels_training labels testing_hists