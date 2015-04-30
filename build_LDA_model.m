%% Trains Latent Dirichlet Allocation model for normal video clips
%
% Input: 
%    data: F x D x V 3-dimensional matrix where F is the number of
% features, D is the dimensionality of the features (how many features), and
% V is the number of videos.
%    C: codebook size
%    L: is the number of topics
%
% Output: 
%    alpha: parameter of Dirichlet distribution
%    beta: parameters for topic distributions
%
% NOTE: to use this model to get the logprob of a clip falling into it, call:
%
%    addpath('fastlda')
%    [~,perplexity,logProb] = applyFastlda(TEST, alpha, beta); % TEST is 1 x D
%    rmpath('fastlda')
%
function [alpha, beta] = build_LDA_model(data, C, L)

[F, D, V] = size(data);

training_data = nan(V, D);
for i = 1:V
    training_data(i,:) = build_clip_histogram(data(:,:,i), C);
end

% random initial alpha/beta, lap is smooth paramter (using default)
oldAlpha = rand(L,1);
oldBeta = rand(L,D);
oldBeta = oldBeta ./ (sum(oldBeta,2)*ones(1,D));
lap = 0.0001;

addpath('fastlda')

[alpha, beta, ~, ~, ~] = learnFastlda(training_data,oldAlpha,oldBeta,lap);

rmpath('fastlda')

end