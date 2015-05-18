%% read text files
videos = {'01', '02', '03', '04', '05', '06',...
    '07', '08', '09', '10', '11'};
transitions = [32, 45, 20, 37, 33, 30,...
    49, 30, 36, 37, 47];
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
        temp2(:, :, 1) = temp1(:, 11 : 25);
        temp2(:, :, 2) = temp1(:, 26 : 40);
        this_clips{j} = sqrt(temp2(:, :, 1) .^ 2 + temp2(:, :, 2) .^ 2);
        clear temp1 temp2;
    end
    clips = [clips; this_clips];
    
    this_labels = ones(num_files, 1);
    this_labels(1 : transitions(i), 1) = 0;
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

%% separate training data
mask_train = (labels(:, 1) == 0) & (indices(:, 1) >= 3) & (indices(:, 1) <= 7);
clips_train = clips(mask_train);
labels_train = labels(mask_train);
clear mask_train;
save 'clips_train' clips_train;
save 'labels_train' labels_train;

%% extract words
k = 10;
features = zeros(length(clips), 15 * k);
for i = 1 : length(clips)
    trajs = clips{i};
    num_trajs = size(trajs, 1);
    if num_trajs > k
        [code_indices, codes] = kmeans(trajs, k);
        code_hist = hist(code_indices, 1 : k);
        [~, code_order] = sort(code_hist);
        codes = codes(fliplr(code_order), :);
    else
        codes = [trajs; repmat(zeros(1, 15), k - num_trajs, 1)];
    end
    features(i, :) = reshape(codes', 1, []);
end
mask_train = (labels(:, 1) == 0) & (indices(:, 1) >= 3) & (indices(:, 1) <= 7);
features_train = features(mask_train, :);
clear i trajs num_trajs codes code_indices code_hist code_order mask_train;
save 'features' features;
save 'features_train' features_train;