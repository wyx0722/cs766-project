# cs766-project
Codebase for our CS 766 Final Project

## Pipeline

*The main function to run tests is "basic_pipeline_function.m".*

1. We extract dense trajectories using the code at http://lear.inrialpes.fr/people/wang/dense_trajectories
2. We assemble the extracted features into a .mat file using "make_full_feature_data.m"
3. We call "basic_pipeline_function.m" for each test of interest. Scripts to call this cuntion are "pipeline_function_cross_val_script.m" and "pipeline_function_orig_paper_script.m".

## Datasets
The dataset is not included to save space. The raw videos for our tests can be found at:

* UMN: http://mha.cs.umn.edu/proj_events.shtml##crowd
* UCF: http://www.vision.eecs.ucf.edu/projects/rmehran/cvpr2009/Abnormal_Crowd.html##VI._Downloads

## all_features.mat
The script "make_full_feature_data.m" processes the dense trajectory output and combines all of the clip features into a single file called "all_features.mat". This file is over 600 MB large for the UMN dataset, so it cannot be hosted on github.

## Explanation of MATLAB scripts:

* _basic_pipeline_function.m_: A function to capture the whole pipeline. This function performs a single run of our pipeline.
* _aggregate_graphs.m_: Build graphs for feature test results
* _aggregate_graphs_kernels.m_: Build graph for multiple kernel test results
* _build_clip_histogram.m_: Computes a histogram for each clip using VQ or LLC coding
* _build_codebook.m_: Computes a codebook with C codewords for a give set of training data
* _compute_interaction_force.m_: Original implementation of social force (computed from raw dense trajectories)
* _make_full_feature_data.m_: Produce a single .mat file containing all features from the raw output of the dense trajectories code
* _pipeline_function_cross_val_script.m_: Run several classification tests for each feature using random training clips. Calls "basic_pipeline_function.m"
* _pipeline_function_orig_paper_script.m_: Run a classification test for each feature using the same training clips as the original paper. Calls "basic_pipeline_function.m"
* _run_classifier.m_: Run the single-class SVM on a set of train/testing data with the specified kernel option
* _social_force_ke.m_: Another implementation of social force