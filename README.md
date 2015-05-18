# cs766-project
Codebase for our CS 766 Final Project

## Pipeline

*The main function to run tests is "basic_pipeline_function.m".*

1. We extract dense trajectories using the code at http://lear.inrialpes.fr/people/wang/dense_trajectories
2. We assemble the extracted features into a .mat file using "make_full_feature_data.m"
3. We call "basic_pipeline_function.m" for each test of interest. Scripts to call this cuntion are "pipeline_function_cross_val_script.m" and "pipeline_function_orig_paper_script.m".