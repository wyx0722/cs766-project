%% Builds codebook
%
% Input: 
%    data: V length cell array where V{i} is a 2-d matrix of F  features, 
%       by D feature dimensions for video i. D should be equal for all i,
%       while F can be different!
%    C: codebook size
%    attempts: number of random restarts to avoid local solutions
%
% Output: 
%    codebook: C x D dictionary matrix
%
function [codebook] = build_codebook(data, C, attempts)

V = length(data);
all_features = [];

for i = 1:V
    all_features = [all_features; data{i}];
end

options = statset('kmeans');
options.Display = 'iter';
options.MaxIter = 200;
options.UseParallel = true;
[~, codebook] = kmeans(all_features, C, 'Replicates', attempts, 'Display', 'iter', 'MaxIter', 200, 'Options', options); % try x times with different centers

end