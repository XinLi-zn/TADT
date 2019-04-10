function [res_map, scale_ind] = cal_scale(res_maps, scale_weights)

% this file selects the most confident scale
% Input:
%       res_maps      -  response maps
%       scale_weights - penalty weight of each scale
% Output:
%       res_map       - response map of the scale
%       scale_ind     - calculated scale index
% By Xin Li, 2017-12-2



% if num_scale == 1
%    res_map = res_maps;
%    scale_ind = 1;
%    return;
% end

num_scale = numel(scale_weights);
res_scale = bsxfun(@times, res_maps, reshape(scale_weights, 1,1, num_scale));

res = reshape(res_scale, [], num_scale);
res_max = max(res);

[~, scale_ind] = max(res_max);
res_map = res_maps(:,:, scale_ind);

end