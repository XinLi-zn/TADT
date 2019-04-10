function [re_image] = imresize_gpu(image,re_param)
%This file implements the imresize function running on GPU based on
%vl_nnbilinearsampler.
% Input:
% image         -      image to be processed, gpuArray
% re_param      -   if its size is one, it is the ratio that to be resized
%               -   if it contains two numbers, it is the resized size.
% Output:
% re_image  -      the resized image.
% By Xin Li, 2018-6-5, UCM

if numel(re_param)==1
    sz = size(image);
    output_sz = round(sz(1:2)*re_param);
else
    output_sz = re_param;
end

yi = linspace(-1, 1, output_sz(1));
xi = linspace(-1, 1, output_sz(2));
[xx,yy] = meshgrid(xi,yi);
yyxx = single([yy(:), xx(:)]') ;
g = reshape(yyxx, 2, output_sz(1), output_sz(2), []);
re_image = vl_nnbilinearsampler(image, gpuArray(g));

end
