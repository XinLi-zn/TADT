% this file calculates the size of the search window based on the 
% input target size and the maximum size
% Input:
%      target_sz    -   the size of the target
%      img_sz       -   the size of the image
%      min_scale    -   the minimum size ratio of the search window to the target default set is 3
% Output:
%      sz           -   the search window size
%      ratio        -   the resize ratio.
% By Xin Li
% 2017-08-29

function [sz, ratio]=cal_window_sz(target_sz, img_sz, min_scale)

sz = target_sz * min_scale;

flag = sz > img_sz;

% the window size on original image
if sum(flag) ==0
    sz = max(sz);
else
    sz = min(img_sz);
end

sz = sz + 8-mod(sz-4,8); % 4*(2k+1)

ratio = sz ./ target_sz;

end