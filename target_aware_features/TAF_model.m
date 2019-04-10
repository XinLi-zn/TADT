
% Select target active features.

% Input:
%   feature_groups    -  a cell array with each cell is one kind of
%                        features of the search image window
%   filter_sz         -  size of the filter [height wieth depth], also the
%                        target size in the feature map.
%   sel_feat_num      -  selected feature numbers

% Output:
%   feat_weight -   weights of each features.
%   IAF_model   -   initialized IAF model

% by Xin Li, April 2 2018
function [feat_weights,w_balance] = TAF_model( feature_groups, filter_sz)

assert(iscell(feature_groups),'the input feature should be a cell array!\n');
feat_num     = numel(feature_groups);
feat_weights = cell(1,feat_num);
channel_num  = [300,80]; % we select 300 and 80 filters from the conv43 and conv41 layer features.
nz_num       = 0;
nz_num_min   = 250;       % the minimum filter number.
w_balance    = gpuArray(single([1 1 1])); % weights leveraging these two features.

%peform feature selection on the conv41 and conv43 layer features.
for layer_i=feat_num:-1:1
    %% initialization
        filter_sz(3) = size(feature_groups{layer_i},3);
        [model.net] = TAF_net_init(filter_sz);
        
        %generate the Gaussian label map
        output_sigma = filter_sz([1:2])*0.1; % output_sigma_factor=0.1
        [h,w,~] = size(feature_groups{layer_i}); sw_sz = [h w]; % search window size [height width]
        model.label_gaussian = single(gaussian_shaped_labels(output_sigma, sw_sz));
        model.label_binary = single(binary_labels(filter_sz([1:2]), sw_sz, [0.3 0.7]));
    %% First train the network with ridge loss
        trainOpts.numEpochs=100;
        trainOpts.learningRate=5e-7;
        trainOpts.weightDecay= 1;
        trainOpts.batchSize = 1 ;
        trainOpts.gpus = 1 ;
        trainOpts.train = 1;
        input = {feature_groups{layer_i}, gpuArray(model.label_gaussian)};

        [net_trained, ~, ders_iter] = cnn_train_dag_ridge(model.net, [], input, [], trainOpts);
        [v,i] = sort(sum(sum(net_trained.params(1).value)),'descend'); % GAP. 
        % The value of the parameters equals to the sum of the gradients in
        % all BP processes. And we found that using the converged parameters is more
        % stable.
    %% mapping gradients values to weights
        feat_weight=zeros(1,1,filter_sz(3), 'like', feature_groups{layer_i});
        feat_weight(i(v>0))=1;     % here we select the filters with positive gradient values, which have a positive relationship with the target loss.
        if layer_i>1 % we perform scale sensitive feature selection on the conv41 feature, as it retains more spatial information.
            filter_sz(3) = size(feature_groups{layer_i},3);
            % initialize the ranking loss network
            [IAF_net_rank, filter_sz_new] = TAF_net_init_rank(filter_sz); 
            % compute the scale sensitive features
            [temp_weight, temp_i] = rank_selection(feature_groups{layer_i}, filter_sz_new, IAF_net_rank);
            % select the features not only are active to the target but also are scale-sensitieve.
            feat_weight = feat_weight.*temp_weight;
        end
        feat_weight(i(channel_num(layer_i)+1:end)) = 0; % we only select the first N (channel_num) filters.
        nz_num=nz_num+sum(feat_weight);    
        
        % In case, there are too less features, we set a minimum feature
        % number. If the total number is less than the minimum number, then select more from conv43
        if layer_i==1 && nz_num<nz_num_min 
            added_inds=(i(sum(feat_weight)+1:sum(feat_weight)+nz_num_min-nz_num));
            feat_weight(added_inds) = 1;
        end
        
        % To combine the features from conv41 and conv43 effective, we define 
        % balance weights based on the maximum value in their features.
        feat_weights{layer_i}=feat_weight;
        w_balance(layer_i) = max(max(sum(feature_groups{layer_i}(:,:,i(1:50)),3)));
        clear net_trained;
        
end
        w_balance=w_balance(1)./w_balance;
end



function [feat_weight, i_index] = rank_selection(feat, filter_sz, net)
% this function selects the scale sensitve features based on the ranking
% loss.

    channel_num= size(feat,3);
    [scale_samples, pair_label, pairs, locs] = gen_ranked_samples(feat, filter_sz(1:2));

% gradients of samples with different scales
    inputs = {'input', gpuArray(scale_samples), 'scale_label', gpuArray(pair_label)};
    net.eval(inputs,{'objective_scale',1});
    feat_der = sum(net.vars(1).der,4);
        
    [v_scale,i_index] = sort(sum(sum(sum(feat_der)),4), 'descend');
    feat_weight=ones(1,1,channel_num, 'like', feat)*0;
    feat_weight(i_index(v_scale>0))=1;  % we select the features, which have positive relationships with the ranking loss.
end

function [IAF_net_rank,filter_sz] = TAF_net_init_rank(filter_sz)
rng('default');

channel=filter_sz(3);
   
rw=ceil(filter_sz(2)/2);
rh=ceil(filter_sz(1)/2);

fw=2*rw+1;
fh=2*rh+1;

filter_sz = [fh,fw,channel];
    
IAF_net_rank=dagnn.DagNN();

%% rank loss net

IAF_net_rank.addLayer('conv12', dagnn.Conv('size', [fw,fh,channel,1],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'input', 'conv12', {'conv21_f', 'conv21_b'});

f = IAF_net_rank.getParamIndex('conv21_f') ;
IAF_net_rank.params(f).value=single(randn(fh,fw,channel,1) /...
    sqrt(rh*rw*channel))/1e8;
IAF_net_rank.params(f).learningRate=1;
IAF_net_rank.params(f).weightDecay=1e3;

f = IAF_net_rank.getParamIndex('conv21_b') ;
IAF_net_rank.params(f).value=single(zeros(1,1));
IAF_net_rank.params(f).learningRate=2;
IAF_net_rank.params(f).weightDecay=1e3;
IAF_net_rank.addLayer('rank_loss',...
    RankingLoss(),{'conv12','scale_label'},'objective_scale');

IAF_net_rank.move('gpu');
end
