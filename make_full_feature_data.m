videoDirectoy = '/media/Data/School/CS766/Final Project/data/features/UMN/';
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
clip_raw_features.SocialForce = {};
clip_raw_features.labels = {};

for i = 1 : num_videos
    files = dir([videoDirectoy videos{i}, '_*.txt']);
    num_files = length(files);
    
    for j = 1 : num_files
        temp_clip = load([videoDirectoy files(j).name]);
        
        clip_raw_features.clipIndexInfo{i} = 1:num_files; % [video #, clip #]
        clip_raw_features.rawFileNames{i,j} = files(j).name;
        clip_raw_features.generalInfo{i,j} = temp_clip(:,genInfoIndices); % i=video, j=clip
        clip_raw_features.denseTrajectories{i,j} = temp_clip(:,denseTrajIndices);
        clip_raw_features.HOG{i,j} = temp_clip(:,HOGIndices);
        clip_raw_features.HOF{i,j} = temp_clip(:,HOFIndices);
        clip_raw_features.MBH{i,j} = temp_clip(:,MBHIndices);
        
        if ( j <= transitions(i) )
            clip_raw_features.labels{i,j} = 1; % normal is a positive example
        else
            clip_raw_features.labels{i,j} = -1; % abnormal is negative
        end
        clear temp_clip
    end
    clear num_files files
end

%social force comp
[~, ~, ~, ~, Sf] = compute_interaction_force(videoDirectoy);
current_index = 0;
for i = 1:length(clip_raw_features.clipIndexInfo)
    for j = clip_raw_features.clipIndexInfo{i}
        current_index = current_index + 1;
        index = clip_raw_features.clipIndexInfo{i}
        clip_raw_features.SocialForce{i,j} = Sf{current_index}(:,:,3);
    end
end
clear Sf
save('all_features.mat',clip_raw_features);