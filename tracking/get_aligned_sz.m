function [ re_ratio, re_t_sz ] = get_aligned_sz( target_sz, max_sz, min_sz )
%calculate the aligned target size
% adjust the re_ratio to make both the height of width of the target be odd
% numbers.


[~, min_axis] = min(target_sz);
max_min_ratio = target_sz(3-min_axis)/target_sz(min_axis);

up_bound = round(sqrt(max_sz^2/max_min_ratio));
low_bound = floor(sqrt(min_sz^2/max_min_ratio));

cand_min_axis = (low_bound:up_bound);
cand_max_axis = cand_min_axis*max_min_ratio;

res_min_axis = mod(cand_min_axis,4);
res_min_axis(res_min_axis>2) = 4-res_min_axis(res_min_axis>2);
res_max_axis = mod(cand_max_axis,4);
res_max_axis(res_max_axis>2) = 4-res_max_axis(res_max_axis>2);

[res_v, ind] = min(mod(res_min_axis,4)*max_min_ratio + mod(res_max_axis,4));

% fprintf('%f\n',res_v(1));

re_ratio = cand_min_axis(ind)/target_sz(min_axis);
re_t_sz = target_sz*re_ratio;
re_t_sz_ori = re_t_sz;
res = mod(re_t_sz,4);
re_t_sz(res<2) = re_t_sz(res<2)-res(res<2);
re_t_sz(res>2) = re_t_sz(res>2)-res(res>2)+4;
re_t_sz=re_t_sz/4;
% fprintf('%f\n',sum(abs(re_t_sz-re_t_sz_ori)));
end

