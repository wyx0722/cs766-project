%% use this function to convert a video clip's raw features into hist gram that is usable in LDA
% INPUTS:
% clipFeatureMatrix is the F x D feature matrix for the video
% codebook is the C x D dictionary
% type is either 'vq' or 'llc'
%
% OUTPUTS
% hist is the 1 x C clip histogram

% this probably didn't need its own function....
function [hist] = build_clip_histogram(clipFeatureMatrix, codebook, type)
    [F, D] = size(clipFeatureMatrix);
    [C, D] = size(codebook);

    data = zeros(F,C);
    hist = nan(1,C);
    
    if strcmp(type,'vq')
        k = 1;
        subdict_ind = knnsearch(codebook,clipFeatureMatrix,'K',k);
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

end