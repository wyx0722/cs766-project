function [intforces] = social_force_ke(clips)

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

end