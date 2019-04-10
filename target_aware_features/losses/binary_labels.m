function labels = binary_labels(target_sz, sw_sz, lu)
% binary_labels generates the binary labels for the search image window
% or_l, or_u is the lower and upper threhold for +1 and -1 class
% Input:
%   target_sz  -    target size in the feature map
%   sw_sz      -    search window size in the feature map
%   lu         -    lower bound and upper bound threholds

% Output:
%   labels     -    generated binary labels, whose size is the same with sw_sz

if nargin<3
    or_l = 0.3;
    or_u = 0.7;
else
    or_l = lu(1);
    or_u = lu(2);
end

sw_w = floor(sw_sz(1)/2);
sw_h = floor(sw_sz(2)/2);
% t_w  = floor(target_sz(1)/2);
% t_h  = floor(target_sz(2)/2);

dist_hor = [sw_w:-1:1 0 1: sw_w];
dist_ver = [sw_h:-1:1 0 1: sw_h];

[rs, cs] = ndgrid( dist_ver , dist_hor);


cs = (target_sz(1) - cs); cs(cs<0) = 0;
rs = (target_sz(2) - rs); rs(rs<0) = 0;

or_map = cs.*rs/prod(target_sz);
or_map = or_map';

labels = zeros(size(or_map)); 
labels(or_map>or_u) = 1;
labels(or_map<or_l) = -1;

end

