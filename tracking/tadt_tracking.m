function [results , fps] = tadt_tracking(img_list, target_loc, vgg16_model_path, display)
% this file is the main interface of the tadt tracker in the paper
% 'Target-Aware Deep Tracking'.

% Input:
%       img_list    -  image paths 
%       target_loc  -  the initial target location 
%       display     -  flag indicating whether show the image result or not
% Output:
%       results     -  tracking results 
%       fps         -  running speed excluding the initialization time (about 2 seconds for a sequence) 
%----------------------By Xin Li, 2018-7-29 

%% parameters initialization
   if nargin < 4
      display = true; 
   end
   img          = imread(img_list{1}); if size(img,3)==1; img = cat(3, img, img, img); end
   
   % ensure the initial target size is not too large or small by resizing
   re_scale = 1; max_size = 59; min_size = 44;
   ori_target_sz = sqrt(prod(target_loc(3:4)));
   
   if ori_target_sz>max_size,  
       re_scale = max_size/ori_target_sz;
   elseif ori_target_sz < min_size
       re_scale = min_size/ori_target_sz;
   end 
   
   img        = imresize(img,re_scale);  
   target_loc = round(target_loc*re_scale);
   
   % initialize scale related parameters
   img_sz       = size(img);   
   scale_num    = 3;
   search_sz    = cal_window_sz(max_size, img_sz([2 1]), scale_num); %
   input_sz     = [search_sz, search_sz];
   switch scale_num
       case 3 
            scales        = [45/47 1 45/43]; % As the feaure map size is 45x45, we use the change step of 2, which benefits center position alignment.
            scale_weights = [0.99 1 1.0055];     
       case 5
            scales        = [1-4/45 1-2/45  1  1+2/45  1 +4/45];
            scale_weights = [0.985 0.988 1 1.005 1.006];
       otherwise error('Undefined scale number!\n');    
   end
   
%% model initialization
   % initialize the vgg16 feature model
   [net_feat, ~] = initVGG16Net(vgg16_model_path);
   % the layers we used.
   feat_layer   = {'relu_43','relu_41'};
   for var_i = 1:numel(feat_layer)
        net_feat.vars(net_feat.getVarIndex(feat_layer{var_i})).precious = true;
   end
   % initialize the siamese matching model
   [net_match] = init_siamese_model();
   
%% First frame processing
   
   % crop the search window and extract its features
   sw_location = floor([ target_loc([1 2])+target_loc([3 4])/2 - search_sz/2, search_sz, search_sz]);
   feat_groups = get_subwindow_feature(net_feat, img, sw_location, input_sz, feat_layer); 
   
   % crop the target exemplar from the feature map
   [target_patches, t_loc] = gene_feat_patch(target_loc([3 4]), sw_location, feat_groups);
   t_loc(3:4)       = t_loc(3:4) - t_loc(1:2)+1; % compute its size  
   
   % handle the size calculation
   sw_feat_sz = size(feat_groups{1});
   if mod(sw_feat_sz(1)*0.05,2)>1
       feat_pad=floor(sw_feat_sz(1)*0.05)+1;
   else
       feat_pad=ceil(sw_feat_sz(1)*0.05)-1;
   end
   assert(mod(feat_pad,2)==0,'pad is not an even number!\n');
   feat_pad=2;
   b_feat_pad = feat_pad/2;
   filter_sz = size(target_patches{1});
    
   % compute the indexes of target-aware features
   [feat_weights,w_balance] = TAF_model(feat_groups, filter_sz);
   % select the target-aware features
   target_patches = feature_selection(target_patches, feat_weights, 'Wreduction',w_balance);  
   patch_template = target_patches{1};
   
  %--show the first frame
   if display    
      figure(2); set(gcf,'Position',[200 300 480 320],'MenuBar','none','ToolBar','none');%axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1]);%
      hd = imshow(img,'initialmagnification','fit'); hold on;
      rectangle('Position', target_loc, 'EdgeColor', [0 0 1], 'Linewidth', 1);    
      set(gca,'position',[0 0 1 1]);
      text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); hold off; drawnow;   
   end
   
   % preparing for the tracking loop
   feat_maps = zeros(size(feat_groups{1},1),size(feat_groups{1},2),size(patch_template,3),scale_num,'like',patch_template);
   num_frame    = numel(img_list);
   results      = zeros(numel(img_list),4);
   results(1,:) = target_loc;
   time=0;
   
%% Tracking loop
   for frame_i = 2 : num_frame
       fprintf('%d\n',frame_i);
       
       % load the new frame
       img = imread(img_list{frame_i});
       tic
       img = imresize(img,re_scale);
       if size(img,3)==1; img = cat(3, img, img, img); end 
        
       % compute its original features
       [feat_groups] = get_subwindow_feature(net_feat, img, sw_location, input_sz, feat_layer);
       
       % compute the target-aware features
       feat_groups = feature_selection(feat_groups, feat_weights, 'Wreduction',w_balance);
       
       % prepare the feature maps with different scales for scale evaluation
       feat_maps(:,:,:,2)=feat_groups{1};
       feats1 = imresize_gpu(feat_maps(:,:,:,2), sw_feat_sz(1:2)+feat_pad );
       feats3 = imresize_gpu(feat_maps(:,:,:,2), sw_feat_sz(1:2)-feat_pad );
       feat_maps(:,:,:,1) = feats1(b_feat_pad+1:end-b_feat_pad,b_feat_pad+1:end-b_feat_pad,:);
       feat_maps(b_feat_pad+1:end-b_feat_pad,b_feat_pad+1:end-b_feat_pad,:,3) = feats3;
       feat_groups{1} = feat_maps;
       
       % perform cross-correlation
       [ res_maps_ori ] = siam_eval( net_match, patch_template, feat_groups);
       
       % estimate the target state based on the above confidence map
       res_map_up = imresize(res_maps_ori, sw_location([4 3]), 'bicubic');
       res_maps = res_map_up+repmat(hann(sw_location(4))*hann(sw_location(3))',1,1,size(res_maps_ori,3));
       [~, scale_ind] = cal_scale(res_maps, scale_weights);
       res_ori = res_maps(:,:,scale_ind);
       [max_h, max_w] = find(res_ori == max(res_ori(:))); %update

       target_loc_old = target_loc;
       t_loc_c = [target_loc(1:2)+target_loc(3:4)/2, target_loc(3:4) ];
       
       t_loc_c(1:2) = t_loc_c(1:2)+ (gather([max_w(1), max_h(1)]) - sw_location(3:4)/2)*scales(scale_ind);
       t_loc_c(3:4) = t_loc_c(3:4) * scales(scale_ind);
       
       target_loc = [t_loc_c(1:2) - t_loc_c(3:4)/2, t_loc_c(3:4)];
       sw_location(3:4) =  round(sw_location(3:4)*scales(scale_ind));
       sw_location(1:2) = [ t_loc_c(1:2) - sw_location(3:4)/2];
       
       % store the results
       results(frame_i,:) = target_loc/re_scale;
       time = time+toc();
%% ------show results
       if display             
           figure(2);
           imagesc(img); hold on;

           % show score map
           xs = floor(target_loc_old(1)+target_loc_old(3)/2) + (1:sw_location(3)) - floor(sw_location(3)/2);
           ys = floor(target_loc_old(2)+target_loc_old(4)/2) + (1:sw_location(4)) - floor(sw_location(4)/2);
           % show results map
           resp_handle = imagesc(xs, ys, imresize(res_maps_ori(:,:,scale_ind),sw_location([4 3]))); colormap hsv;
           alpha(resp_handle, 0.4);
           
           % show bounding box
           rectangle('Position', target_loc, 'EdgeColor', [ 0 1 0], 'Linewidth', 1);  
           set(gca,'position',[0 0 1 1]);
           text(10,10,num2str(frame_i),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
           
           hold off;  drawnow;
       end 
        
   end

fps= num_frame/time;
end

