function [pair_label] = generate_pair_label(label)
% this file converts the labels of a set of scale labels (from the smallest candidate to the largest one) into pair-wise
% labels.

num_scale = numel(label);
base_ind = ceil(num_scale/2);

pair_label = [2:base_ind, base_ind:num_scale-1;...
               1:base_ind-1, base_ind+1:num_scale];

end