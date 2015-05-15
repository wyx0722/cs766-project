clear
videoDirectoy = 'D:\School\CS766\Final Project\data\features\UMN\';
videos = {'01', '02', '03', '04', '05', '06',...
    '07', '08', '09', '10', '11'};
transitions = [32, 45, 20, 37, 33, 30,...
    49, 30, 36, 37, 47];
num_videos = length(videos);

% indices to split up rows in raw files
genInfoIndices = 2:10; % leave out frameNum
denseTrajIndices = 11:42;
HOGIndices = 43:138;
HOFIndices = 139:246;
MBHIndices = 247:438;

% store everything in a giant data file
clip_raw_features.clipIndexInfo = {}; % [video #, clip #]
clip_raw_features.rawFileNames = {};
clip_raw_features.generalInfo = {};
clip_raw_features.denseTrajectories = {}; % 1:16 -> x, 17:32 -> y
clip_raw_features.HOG = {};
clip_raw_features.HOF = {};
clip_raw_features.MBH = {}; % 1:96 -> x, 97:192 -> y
clip_raw_features.SocialForce_Mike = {};
clip_raw_features.DesiredVelocity = {};
clip_raw_features.DesiredVelocityMag = {};
clip_raw_features.ActualVelocity = {};
clip_raw_features.ActualVelocityMag = {};
clip_raw_features.labels = {};

labels = [];
trajectories = {};

%store
current_index = 0;
for i = 1 : num_videos
    files = dir([videoDirectoy videos{i}, '_*.txt']);
    num_files = length(files);
    clip_raw_features.clipIndexInfo{i} = 1:num_files; % length = # videos, clipIndexInfo{i} = array of clip numbers
    
    for j = 1 : num_files
        current_index = current_index + 1;
        temp_clip = load([videoDirectoy files(j).name]);
        
        clip_raw_features.rawFileNames{current_index} = files(j).name;
        clip_raw_features.generalInfo{current_index} = temp_clip(:,genInfoIndices); % i=video, j=clip
        clip_raw_features.denseTrajectories{current_index} = temp_clip(:,denseTrajIndices);
        clip_raw_features.HOG{current_index} = temp_clip(:,HOGIndices);
        clip_raw_features.HOF{current_index} = temp_clip(:,HOFIndices);
        clip_raw_features.MBH{current_index} = temp_clip(:,MBHIndices);
        
        if ( j <= transitions(i) )
            clip_raw_features.labels{current_index} = 1; % normal is a positive example
        else
            clip_raw_features.labels{current_index} = -1; % abnormal is negative
        end
        labels = [labels; clip_raw_features.labels{current_index} i ];
        trajectories{current_index}(:,:,1) = clip_raw_features.denseTrajectories{current_index}(:,1:16);
        trajectories{current_index}(:,:,2) = clip_raw_features.denseTrajectories{current_index}(:,17:32);
        clear temp_clip
    end
    clear num_files files
end
%clear Sf

%social force comp
[actualVel, desiredVel, ~, ~, Sf] = compute_interaction_force(videoDirectoy, trajectories, labels);
clear labels trajectories
for cInd = 1:length(clip_raw_features.labels)
    clip_raw_features.SocialForce_Mike{cInd} = Sf{cInd}(:,:,3);
    clip_raw_features.DesiredVelocity{cInd} = [desiredVel{cInd}(:,:,1) desiredVel{cInd}(:,:,2)];
    clip_raw_features.DesiredVelocityMag{cInd} = sqrt( (desiredVel{cInd}(:,:,1)).^2 + (desiredVel{cInd}(:,:,2)).^2 );
    clip_raw_features.ActualVelocity{cInd} = [actualVel{cInd}(:,:,1) actualVel{cInd}(:,:,2)];
    clip_raw_features.ActualVelocityMag{cInd} = sqrt( (actualVel{cInd}(:,:,1)).^2 + (actualVel{cInd}(:,:,2)).^2 );
end

save('all_features.mat','clip_raw_features');