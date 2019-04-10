% this file extracts the specified deep features of the input images with
% the input net
% Input:
%      net_feat   -  net used to extract features
%      img        -  image to be processed
%      locs  -  the center (target) position and the cropped patch 
%                    [ x y w h], can contain several locations
%      input_sz   -  the input size of the feature extraction nets. [w h] 
%      layer_name -  the name of the layer whose feature is to be used
% Output:
%       feats     -  the extracted features of the cropped image patch
% By Xin Li
% 2017-8-30
function [feat_group, patch_sw] = get_subwindow_feature(net_feat, img, locs, input_sz, layer_name) 

    num_layers = numel(layer_name);
    num_locs    = size(locs,1);
    
    img = single(img)-128;
    feat_group = cell(1,num_layers);
    
    if num_locs>1
        locs = mat2cell(locs,ones(1,num_locs),4);
        
        patch_sw = cellfun(@(loc) get_subwindow(img, loc([2 1])+loc([4 3])/2, loc([4 3])),...
                                locs, 'uniformoutput', false);% pos should be the center of the target (y,x)
        patch_sw = cellfun(@(patch) imresize(patch, [input_sz(2) input_sz(1)],'bilinear','antialiasing',false),...
                                patch_sw,'uniformoutput', false);
        patch_sw = cell2mat( reshape(patch_sw,1,1,1,num_locs) );
    else
        patch_sw = get_subwindow(img, locs([2 1])+locs([4 3])/2, locs([4 3]) ); % pos should be the center of the target (y,x)
       
        patch_sw = imresize(patch_sw, [input_sz(2) input_sz(1)],'bilinear','antialiasing',false);
        
    end
    
    net_feat.eval({'input',gpuArray(patch_sw)});
    
        for j=1:num_layers

            feat = net_feat.vars(net_feat.getVarIndex(layer_name{j})).value;

            feat_group{j} = feat;
        end

end