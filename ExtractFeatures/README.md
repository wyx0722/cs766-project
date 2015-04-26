# A Handy Program to Clip Videos and Extract Features

## Introduction

* Clip the input video every 15 frames.

* Extract dense trajectories and trajectory-aligned features (HOF, HOG, MBH) for video clips.

* Save the extracted features into text files.

## Compilation

Type `make`.

## Usage

Type `./release/ExtractFeatures [video_file_name] [feature_file_prefix]`.

(e.g. `./release/ExtractFeatures ./videos/test.mp4  ./features/test`)

Ensure that the output folder has already been created before running the program.

## Output Format

The features are computed one by one, and each one in a single line, with the following format:

`frameNum mean_x mean_y var_x var_y length scale x_pos y_pos t_pos Trajectory HOG HOF MBHx MBHy`

The first 10 elements are information about the trajectory:

* frameNum:     The trajectory ends on which frame
* mean_x:       The mean value of the x coordinates of the trajectory
* mean_y:       The mean value of the y coordinates of the trajectory
* var_x:        The variance of the x coordinates of the trajectory
* var_y:        The variance of the y coordinates of the trajectory
* length:       The length of the trajectory
* scale:        The trajectory is computed on which scale
* x_pos:        The normalized x position w.r.t. the video (0~0.999), for spatio-temporal pyramid 
* y_pos:        The normalized y position w.r.t. the video (0~0.999), for spatio-temporal pyramid 
* t_pos:        The normalized t position w.r.t. the video (0~0.999), for spatio-temporal pyramid

The following element are five descriptors concatenated one by one:

* Trajectory:    2x[trajectory length] | (default 30 dimension) 
* HOG:           8x[spatial cells]x[spatial cells]x[temporal cells] | (default 96 dimension)
* HOF:           9x[spatial cells]x[spatial cells]x[temporal cells] | (default 108 dimension)
* MBHx:          8x[spatial cells]x[spatial cells]x[temporal cells] | (default 96 dimension)
* MBHy:          8x[spatial cells]x[spatial cells]x[temporal cells] | (default 96 dimension)

**Note: Trajectory is represented by a sequence of normalized point coordinates, which is different from the original DenseTrack code.**
