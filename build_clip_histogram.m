%% use this function to convert a video clip's raw features into hist gram that is usable in LDA
% clipFeatureMatrix is the F x D feature matrix for the video and C is the
% codebook size

% this probably didn't need its own function....
function [hist] = build_clip_histogram(clipFeatureMatrix, C)
    [~,clusters] = kmeans(clipFeatureMatrix,C);
    
    % sum pool the clusters (maybe something better?)
    hist = sum(clusters);
end