load('clips.mat');
load('labels.mat');
load('indices.mat');

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
mask_train = (labels(:, 1) == 0) & (indices(:, 1) >= 3) & (indices(:, 1) <= 7);
intforces_train = intforces(mask_train);
trajs_train = zeros(0, 14);
for i = 1 : sum(mask_train)
    trajs_train = [trajs_train; sqrt(intforces_train{i}(:, :, 1) .^ 2 + intforces_train{i}(:, :, 2) .^ 2)];
end
[~, codebook] = kmeans(trajs_train, 1024, 'Display', 'iter', 'MaxIter', 200);
clear trajs_train i mask_train intforces_train;
save 'codebook' codebook;

%% extract words
features = zeros(length(intforces), 1024);
for i = 1 : length(intforces)
    trajs = sqrt(intforces{i}(:, :, 1) .^ 2 + intforces{i}(:, :, 2) .^ 2);
    words = knnsearch(codebook, trajs);
    features(i, :) = hist(words, 1 : 1024);
    features(i, :) = features(i, :) / sum(features(i, :));
end
mask_train = (labels(:, 1) == 0) & (indices(:, 1) >= 3) & (indices(:, 1) <= 7);
features_train = features(mask_train, :);
clear i trajs words mask_train;
save 'features' features;
save 'features_train' features_train;