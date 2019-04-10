function [patches,loc] = gene_feat_patch( target_sz, sw_loc, feat_groups )
% this function crops the features of the target
% Input:
%       sw_loc      - location of the search window
%       target_sz   - size of the target
% Output:
%       patches     - cell array
%       loc         - target position in the feature map.
% By Xin Li, 11-28-2017

num_feats = numel(feat_groups);
patches = cell(1, num_feats);

for i=1:num_feats
    feat_map = feat_groups{i};
    size_feat = size(feat_map);
    center = round(size_feat([2 1])/2);
    re_t_sz = floor(target_sz .* size_feat([2 1])./sw_loc([3 4])/2)*2+1;
    loc = [ center-floor(re_t_sz/2) center+floor(re_t_sz/2)];
    
    patches{i} = feat_map(loc(2):loc(4), loc(1):loc(3), :,:);
end
    

end