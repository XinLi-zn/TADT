function [samples, pair_label, pairs, locs] = gen_ranked_samples(feat_map, target_sz)
% this function generates samples with different scales and offsets based
% on the input target position. (make the size ratio from 0.5 to 2 as
% possible)
% Input: 
%   feat_map - input feature whose size is three times of the target size.
%   target_sz - position of the target.
% 
% Output:
%   samples - samples with different scales
%   pair_label - label of scaled samples
%   pairs    - samples with different offsets
%   locs   -  sample locations [tl_x tl_y, br_x, br_y] 

if iscell(feat_map)
    feat_map = feat_map{1};
end

map_sz = size(feat_map);
target_loc = [floor(map_sz(1:2)/2)+1 target_sz]; %[y x height width ]

%% generate samples (all the position with the same size with target_loc)
target_pad = floor(target_sz/2);
ys1 = (target_pad(1)+1:map_sz(1)-target_pad(1));
xs1 = (target_pad(2)+1:map_sz(2)-target_pad(2));
[yy1,xx1] = meshgrid(ys1,xs1);
grid_num = numel(yy1(:));

locs1 = [xx1(:) yy1(:) ones(grid_num,1)*target_sz(2) ones(grid_num,1)*target_sz(1)];
locs_tmp = locs1;
r = overlap_ratio(locs1, target_loc([2 1 4 3])); % rect1 = [c_X c_Y Width Height]

% label generation. Three sets are used set-1 (r = [0.9 1]), set-2 (r=[0.6 0.9]), set-3 (r=[0.3 0.6]), set-4 (r = 0)
set_thres = [0.9 0.6 0.3 -0.1];
p_num = sum(r>set_thres(3));
n_num_ori = grid_num - p_num;
rm_num = max(n_num_ori-3*p_num,0);

[r_v_ori,r_i_ori] = sort(r,'descend');
n_inds = r_i_ori(p_num+1:end);
locs_tmp(n_inds(randperm(end, rm_num)),:)=[];

r = overlap_ratio(locs_tmp, target_loc([2 1 4 3]));
[r_v_ori,r_i_ori] = sort(r,'descend');

set_inds = cell(1,4);
for seti = 1 : numel(set_thres)
     set_inds{seti} = r_i_ori( r_v_ori>set_thres(seti) );
     r_v_ori(r_v_ori>set_thres(seti))=-1;
end

% generate label pairs, the number is 3 times of the larger number.
repeat_num = 3;
pairs1 = vec2matrix(set_inds{1},set_inds{2},repeat_num);
pairs2 = vec2matrix(set_inds{2},set_inds{3},repeat_num);
pairs3 = vec2matrix(set_inds{3},set_inds{4},repeat_num);
pairs4 = vec2matrix(set_inds{1},set_inds{4},repeat_num);
pairs5 = vec2matrix(set_inds{2},set_inds{4},repeat_num);
pairs = [pairs1,pairs2,pairs3,pairs4,pairs5];

inds = [ set_inds{1}; set_inds{2}; set_inds{3}; set_inds{4}];
locs = crop_feat(feat_map, locs_tmp(inds,:));

%% generate samples (2. overlap ratio with gt >0.1 with different scales from 0.5 to 2)
m_c_h = floor(map_sz(1)/2)+1;
% m_c_w = floor(map_sz(2)/2)+1;
assert(prod(mod(target_sz,2))==1, 'target size is not an odd number!\n');
b_t_sz = floor(target_sz/2);

re_szs = (max(b_t_sz): 2*(m_c_h-1))*2+1;
c_ind = find(re_szs == map_sz(1));
l_pad = c_ind-1;
r_pad = map_sz(1)-c_ind;
pad = min([l_pad,r_pad]);
re_szs = re_szs(c_ind-pad:c_ind+pad);

ratios = re_szs/map_sz(1);

t_locs = mat2cell([-b_t_sz(1) + floor((re_szs-1)/2)+1; b_t_sz(1) + floor((re_szs-1)/2)+1;...
                   -b_t_sz(2) + floor((re_szs-1)/2)+1; b_t_sz(2) + floor((re_szs-1)/2)+1],4,ones(1,numel(re_szs)) );

re_szs_cell = num2cell(re_szs);
re_feats = cellfun(@(re_sz) imresize(gather(feat_map), [re_sz, re_sz]),...
     re_szs_cell, 'uniformoutput', false);
t_feats = cellfun(@(feat, t_loc) feat(t_loc(1):t_loc(2), t_loc(3):t_loc(4),:), re_feats, t_locs, 'uniformoutput', false);
samples = cell2mat(reshape(t_feats, 1,1,1,numel(re_szs)));

labels = 1-(ratios-1).^2;
[pair_label] = generate_pair_label(labels);
end

function [pair_label] = generate_pair_label(label)
% this file converts the labels of a set of scale labels (from the smallest candidate to the largest one) into pair-wise
% labels.

num_scale = numel(label);
base_ind = ceil(num_scale/2);

pair_label = [2:base_ind, base_ind:num_scale-1;...
               1:base_ind-1, base_ind+1:num_scale];

end

function [locs] = crop_feat(feat_map, locs_ori)
% locs = [c_X c_Y Width Height]
    pad = floor(locs_ori(1,3:4)/2);
    locs(:,1) = locs_ori(:,1) - pad(1);
    locs(:,2) = locs_ori(:,2) - pad(2);
    locs(:,3) = locs_ori(:,1) + pad(1);
    locs(:,4) = locs_ori(:,2) + pad(2);
    
%     locs = mat2cell(locs, ones(size(locs,1),1),4);
%     t_feats = cellfun(@(loc) feat_map(loc(2):loc(4), loc(1):loc(3),:), locs, 'uniformoutput', false);

end

function pairs =vec2matrix(vec1,vec2,repeat_num)

num1 = numel(vec1);
num2 = numel(vec2);

ratio = floor(num2/num1);
left_num = num2-ratio*num1;
vec1s = repmat(vec1,ratio,1);
vec1s = [vec1s(:); vec1(randperm(end, left_num))];
pairs = [vec1s';vec2'];

end

function r = overlap_ratio(rect1, rect2)
% rect1 = [X Y Width Height]
inter_area = rectint(rect1,rect2);
union_area = rect1(:,3).*rect1(:,4) + rect2(:,3).*rect2(:,4) - inter_area;

r = inter_area./union_area;
end