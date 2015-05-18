%% read text files
videos = {'01', '02', '03', '04', '05', '06',...
    '07', '08', '09', '10', '11', '12',...
    '13', '14', '15', '16', '17', '18',...
    '19', '20'};
transitions = 12;
num_videos = length(videos);
clips = cell(0, 1);
labels = zeros(0, 1);
indices = zeros(0, 1);
for i = 1 : num_videos
    files = dir([videos{i}, '_*.txt']);
    num_files = length(files);
    
    this_clips = cell(num_files, 1);
    for j = 1 : num_files
        temp1 = load(files(j).name);
        temp2(:, :, 1) = temp1(:, 11 : 26);
        temp2(:, :, 2) = temp1(:, 27 : 42);
        this_clips{j} = temp2;
        clear temp1 temp2;
    end
    clips = [clips; this_clips];
    
    this_labels = ones(num_files, 1);
    this_labels(:) = i > transitions;
    labels = [labels; this_labels];
    
    this_indices = ones(num_files, 1);
    this_indices(:, 1) = i;
    indices = [indices; this_indices];
end
labels = logical(labels);
indices = uint8(indices);
clear videos transitions num_videos i j files num_files this_clips this_labels this_indices;
save 'clips' clips;
save 'labels' labels;
save 'indices' indices;

%% compute interaction force
relaxation = 1;
K = 5;
intforces = cell(length(clips), 1);
for i = 1 : length(clips)
    frames = clips{i};
    num_trajs = size(frames, 1);
    num_frames = size(frames, 2);
    velocities = zeros(num_trajs, num_frames - 2, 2);
    for j = 1 : num_frames - 2
        velocities(:, j, :) = (frames(:, j + 2, :) - frames(:, j, :)) / 2;
    end
    accelerations = zeros(num_trajs, num_frames - 2, 2);
    for j = 1 : num_frames - 2
        accelerations(:, j, :) = frames(:, j + 2, :) - frames(:, j + 1, :) * 2 + frames(:, j, :);
    end
    avg_velocities = zeros(num_trajs, num_frames - 2, 2);
    for j = 1 : num_frames - 2
        knn_idx = knnsearch(squeeze(frames(:, j + 1, :)), squeeze(frames(:, j + 1, :)), 'K', K);
        avg_velocity = zeros(num_trajs, 2);
        for k = 1 : num_trajs
            avg_velocity(k, :) = sum(velocities(knn_idx(k, :), j, :), 1) / K;
        end
        avg_velocities(:, j, :) = avg_velocity;
    end
    intforces{i} = (avg_velocities - velocities) / relaxation - accelerations;
end
clear accelerations avg_velocities avg_velocity frames i j k knn_idx num_frames num_trajs velocities;
save 'intforces' intforces;

%% construct dictionary
intforces_train = intforces(ismember(indices, 1 : 12));
trajs_train = zeros(0, 14);
for i = 1 : sum(mask_train)
    trajs_train = [trajs_train; sqrt(intforces_train{i}(:, :, 1) .^ 2 + intforces_train{i}(:, :, 2) .^ 2)];
end
[~, codebook] = kmeans(trajs_train, 1024, 'Display', 'iter', 'MaxIter', 400);
clear trajs_train i intforces_train;
save 'codebook' codebook;

%% extract words
features = zeros(length(intforces), 1024);
for i = 1 : length(intforces)
    trajs = sqrt(intforces{i}(:, :, 1) .^ 2 + intforces{i}(:, :, 2) .^ 2);
    words = knnsearch(codebook, trajs);
    features(i, :) = hist(words, 1 : 1024);
    features(i, :) = features(i, :) / sum(features(i, :));
end
clear i trajs words;
save 'features' features;