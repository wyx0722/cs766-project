%% Trains Latent Dirichlet Allocation model for normal video clips
%
% Input: 
%    data: V length cell array where V{i} is a 2-d matrix of F  features, 
%       by D feature dimensions for video i. D should be equal for all i,
%       while F can be different!
%    codebook: C x D matrix for the dictionary
%    L: is the number of topics
%    codingType: should be 'vq' or 'llc'
%
% Output: 
%    alpha: parameter of Dirichlet distribution
%    beta: parameters for topic distributions
%
function [alpha, beta] = build_LDA_model(data, codebook, L, codingType)

V = length(data);
[C, D] = size(codebook)

training_data = nan(V, C);
for i = 1:V
    training_data(i,:) = build_clip_histogram(data{i}, codebook, codingType);
end

% Run LDA (need to adjust to use lda-c)

end