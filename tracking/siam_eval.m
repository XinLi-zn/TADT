function [ res_map ] = siam_eval( net_match, patches, feat_groups )
%DSIAM_EVAL This function evaluates the score of each positions on the
%search map
% Input:
%       net_match   - matching net
%       patches     - feature groups of the target
%       feat_groups - feature groups of the search image patch
%       sw_sz       - size of the search window [width hight]
% Output:
%       res_map     - response map
% By Xin Li, 2017-12

    for feat_i =1 : numel(feat_groups)

        [x_h, x_w, ~, num_scale] = size(feat_groups{feat_i});
        [z_h, z_w,~, num_scale_1] = size(patches); 
        num_scale = max([num_scale,num_scale_1]);
        z_bh = floor(z_h/2); z_bw = floor(z_w/2);
        res_tmp = zeros(x_h, x_w, num_scale,'like',feat_groups{feat_i});

        net_match.eval({'w_feat', patches,'x_feat', feat_groups{feat_i}});
        res_tmp(z_bh+1:z_bh+x_h-z_h+1, z_bw+1:z_bw+x_w-z_w+1,:) = ...
            squeeze( net_match.vars(net_match.getVarIndex('m_score')).value);
        res_ = reshape(gather(res_tmp),x_h,x_w,[]);
        
       
        res_map = res_/max(res_(:));
     
    end
       
end

