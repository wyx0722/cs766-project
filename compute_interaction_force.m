function [vi, viq, ai, f_int, Sf] = compute_interaction_force(videoDirectoy, trajectories, labels)
% compute interaction force

% from the social force paper they use
% Fint = (1/tau)(viq-vi)-dvi/dt
% tau = scalar not provided in the paper
% viq = desired particle velocity = (1-pi)O(xi,yi)+pi(Oave(xi,yi)
% pi = panic weight 0..1 (0 for normal scene)
% vi = Oave(xi,yi) actual velocity of the particle
% O(xi,yi) = optical flow at pixel (xi,yi)
% Oave(xi,yi) = average optical flow at pixel (xi,yi)

% trajectories is a cell array with 1 entry for each file - these files 
% represent trajectories pulled from a sequence of 15 frames of the video.
% each cell contains a 3d matrix the rows represent the individual 
% trajectories, the columns represent the x or y positions for the
% corresponding frames and the thrid dimension is either a 1 or a 2
% corresponding to either the x or y coordinate.
%load([videoDirectoy 'labels.mat'], 'labels');

% labels is a 510x2 matrix each row represents one of the files the first
% column has either a -1 or a 1.  -1 represents normal behavior while +1 
% represents abnormal behavior and the second column represents which
% video this data comes from.
%load([videoDirectoy 'trajectories.mat'], 'trajectories');

% the data structures for the velocities, vi and viq, and the acceleration,
% ai follow the same structure as used by trajectories

% parameters
tau=1;
numfiles = length(labels);
%numfiles=6;  % for debugging use small number of files
frames_per_clip=16;
k_nearest_points=5;
temporal_range=2; % how many frames on either side of the current frame are included


% compute desired particle velocity, viq, assume pi=0 (no panic)
viq = cell(numfiles,1);  
parfor i=1:numfiles  
   trajectory=trajectories{i};
   v = zeros(size(trajectory,1),frames_per_clip-1,2);
   for f=1:frames_per_clip-1
      v(:,f,:)=trajectory(:,f+1,:)-trajectory(:,f,:);
   end
   viq{i}=v;
end


% compute acceleration
ai = cell(numfiles,1);  
parfor i=1:numfiles  
   v=viq{i};
   a = zeros(size(v,1),frames_per_clip-2,2);
   for f=1:frames_per_clip-2
      a(:,f,:)=v(:,f+1,:)-v(:,f,:);
   end
   ai{i}=a;
end


% compute actual (average) particle velocity, vi
vi=cell(numfiles,1);
parfor i=1:numfiles
    trajectory=trajectories{i};
    v=viq{i};
    vi_holder=zeros(size(v,1),frames_per_clip-1,2);
    for frame=1:(frames_per_clip-1)
       points=[trajectory(:,frame,1) trajectory(:,frame,2)];
       knn=knnsearch(points, points, 'K',k_nearest_points);
       which_frames = max(1,frame-temporal_range):min(frame+temporal_range,frames_per_clip-1);
       v_x_list=[];
       v_y_list=[];
       for k=1:size(trajectory,1)
           tv_x=[];
           tv_y=[];
           for j=which_frames
              temp = v(knn(k,:),j,:);
              tv_x=[tv_x temp(:,:,1)'];
              tv_y=[tv_y temp(:,:,2)'];
           end
           v_x_list=[v_x_list; tv_x];
           v_y_list=[v_y_list; tv_y];
       end
       vi_holder(:,frame,1)=sum(v_x_list,2) ./ size(v_x_list,2);
       vi_holder(:,frame,2)=sum(v_y_list,2) ./ size(v_y_list,2);
    end
    vi{i}=vi_holder;
end

% compute interaction force, f_int
    % assume m=1 like in the paper
    % using tau = 1 as set at the top in parameters section
f_int=cell(numfiles,1);
parfor i=1:numfiles
    temp=viq{i}-vi{i};
    temp=temp(:,1:frames_per_clip-2,:)./tau
    f_int{i}=temp-ai{i};
end


% compute magnitude of interaction force vectors at particle positions
% Force_Flow, Sf, is described in the paper as having the same resolution
% as the image.  Here I'm just calculating the magnitude of the force at
% the location of the corresponding particle trajectory
% Sf will be a cell array with a cell for each file where each cell entry 
% is a matrix with dimension 1 corresponding to particles, dimension 2
% corresponding to frame and dimension 3 indicating 1=x position, 2=y
% position, 3=magnitude of force
Sf=cell(numfiles,1);
parfor i=1:numfiles
    trajectory=trajectories{i};
    f=f_int{i}
    temp = trajectory(:,1:frames_per_clip-2,:);
    fx2=f(:,:,1).*f(:,:,1);
    fy2=f(:,:,2).*f(:,:,2);
    temp(:,:,3)=(fx2+fy2).^(0.5);
    Sf{i}=temp
end

