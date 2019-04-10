% construct the Target Active Feature (TAF) network
% This net exploits the ridge loss and the hinge loss to calculate target
% active features.

% Input:
%  filter_sz    -   filter size [height width depth], target size in the
%                   input feature map
% Output:
%  IAF_net      -   the initialized IAF network

% By Xin Li, April 8 2018

function [TAF_net,filter_sz] = TAF_net_init(filter_sz)
rng('default');

channel=filter_sz(3);
    
rw=ceil(filter_sz(2)/2);
rh=ceil(filter_sz(1)/2);

fw=2*rw+1;
fh=2*rh+1;

filter_sz = [fh,fw,channel];
    
TAF_net=dagnn.DagNN();

%% conv layer 1-1 for regression  
TAF_net.addLayer('conv11', dagnn.Conv('size', [fw,fh,channel,1],...
    'hasBias', true, 'pad',...
    [rh,rh,rw,rw], 'stride', [1,1]), 'input', 'conv11', {'conv11_f', 'conv11_b'});

f = TAF_net.getParamIndex('conv11_f') ;
TAF_net.params(f).value=single(randn(fh,fw,channel,1) /...
    sqrt(rh*rw*channel))/1e8;
TAF_net.params(f).learningRate=1;
TAF_net.params(f).weightDecay=1e3;

f = TAF_net.getParamIndex('conv11_b') ;
TAF_net.params(f).value=single(zeros(1,1));
TAF_net.params(f).learningRate=2;
TAF_net.params(f).weightDecay=1e3;

%%
TAF_net.addLayer('L2Loss',...
    RegressionL2Loss(),{'conv11','label_gaussian'},'objective_r');

% for the scale sensitive features, we first select the discriminative
% features with the regression loss then we select the scale sensitive from
% them with the ranking loss. The ranking loss is constructed in the file
% TAF_model.m

end
