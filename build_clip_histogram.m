%% use this function to convert a set of video clips' raw features into histograms that are usable
% INPUTS:
% clipFeatureMatrix is a V length cell array where V{i} is a 2-d matrix of F  features, 
%       by D feature dimensions for video i. D should be equal for all i,
%       while F can be different!
% codebook is the C x D dictionary
% type is either 'vq' or 'llc'
%
% OUTPUTS
% hist is the 1 x C clip histogram

% this probably didn't need its own function....
function [final_matrix] = build_clip_histogram(clipData, codebook, type)
V = length(clipData);
[C, D] = size(codebook);

final_matrix = nan(V,C);
for clip = 1:V
    clipFeatureMatrix = clipData{clip};
    [F, D] = size(clipFeatureMatrix);

    data = zeros(F,C);
    hist = nan(1,C);
    
    if strcmp(type,'vq')
        k = 1;
        subdict_ind = knnsearch(codebook,clipFeatureMatrix,'K',k, 'Distance', 'seuclidean');
        for i = 1:F
            curr_subdict_ind = subdict_ind(i,:);
            data(i,curr_subdict_ind) = 1.0; % single neighbor per row
        end
        
        hist(:) = sum(data);
    elseif strcmp(type,'llc')
        k = 5;
        
        % LLC (aproximate solution as in Section 3)
        subdict_ind = knnsearch(codebook,clipFeatureMatrix,'K',k);
        one_vec = ones(k,1);
        for i = 1:F
            curr_feature = clipFeatureMatrix(i,:);
            curr_subdict_ind = subdict_ind(i,:);
            curr_subdict = codebook(curr_subdict_ind,:);

            %solve LLC for xi
            B_1x = curr_subdict - one_vec * curr_feature;
            C = B_1x * B_1x';
            c_hat = C \ one_vec;
            c_hat = c_hat / sum(c_hat);

            data(i,curr_subdict_ind) = c_hat';
        end
        
        hist(:) = max(data);
    else
        error('Valid types are vq or llc');
    end
    
    final_matrix(clip,:) = hist(1,:);
end
end