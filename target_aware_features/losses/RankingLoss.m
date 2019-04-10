classdef RankingLoss < dagnn.Loss
    % For details, please refer to paper 'Improving pairwise ranking for multi-label image classification'
    % l = log( 1 + sum(exp(f1-f2)) )   (f1 and f2 denote all pairs and f2 ranks higher than f1)


  methods
    function outputs = forward(obj, inputs, params)%inputs is in the order from the smallest size to the highest.
      
%       sam_num = size(inputs{1},4);
%       base_ind = ceil(sam_num/2);
% 
%       pair_inds = [2:base_ind, base_ind:sam_num-1;...
%                    1:base_ind-1, base_ind+1:sam_num]; 
      pair_inds = inputs{2};
      pre_vals = inputs{1}(:)';
      pair_val = pre_vals(pair_inds);  % 2 x pair_num 
      loss_lsep =log(1+ exp(pair_val(1,:) - pair_val(2,:)));
      outputs{1} = loss_lsep;
      n = obj.numAveraged ;      
      m = n + size(pair_inds,2);
      obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
%       mass = sum(sum(inputs{2} > 0,2),1) + 1 ;
      pair_inds = inputs{2};
      pre_vals = inputs{1}(:)';
      pair_val = pre_vals(pair_inds);  % 2 x pair_num 
      loss_lsep =log(1+ sum(exp(pair_val(1,:) - pair_val(2,:))));
      
      sam_num = numel(pre_vals);
      tmp_sum = zeros(1,sam_num); u = tmp_sum; v = tmp_sum;
      for k=1:size(pair_val,2)
          u(:)=0; v(:)=0;
          u(pair_inds(1,k))=1;
          v(pair_inds(2,k))=1;
          delta_y = u-v;
          tmp_sum = tmp_sum + delta_y.*exp(delta_y.*(-pre_vals));
      end
      derInputs{1} = reshape(-1/loss_lsep*(tmp_sum), size(inputs{1}));
      
       derInputs{2} = gpuArray(zeros(size(inputs{2}),'single'));
%       derInputs{2}(obj.crop(1)+1:end-obj.crop(2), obj.crop(3)+1:end-obj.crop(4),:,:) = -derInputs{1};
      derParams = {} ;
    end

    function obj = RegressionL2Loss(varargin)
      obj.load(varargin) ;
    end
    
  end
end
