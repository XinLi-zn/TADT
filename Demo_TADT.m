
clear all;

%Adding necessary paths
addpath(genpath('./tracking'));
addpath(genpath('./utils'));
addpath(genpath('./target_aware_features'))
%% video selection

base_path  = './sequences/'; % base_path
video_path = choose_video(base_path);
video      = video_path(length(base_path)+1:end-1);
[seq_info] = load_seq_info(base_path,video);

%% model path
% Please setting the following model path and then comment the following line
error('Please first set the imagenet-vgg-verydeep-16.mat model path! Or you can download it from www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat')
vgg16_model_path = '.../imagenet-vgg-verydeep-16.mat';

%% Call the main tracking file

display = 1;
[result, fps]=tadt_tracking(seq_info.img_list, seq_info.gt(1,:), vgg16_model_path, display);

%% Calculate the AUC score and running speed and show them on the screen

[distance_precision, area, average_center_location_error] = ...
    compute_performance_measures(result, seq_info.gt);

fprintf('\n*o*--------------Performance--------------*o*');
fprintf('\n   The average overlap ratio is %f',area);
fprintf('\n   The running speed is %f FPS\n', fps);
