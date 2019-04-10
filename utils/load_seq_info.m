function [seq_info]=load_seq_info(base_path,seq_name)
%This function loads the information of the video corresponding to the
%given path seq_path, which is modified from
    % GENCONFIG
    % Generate a configuration of a sequence
    % 
    % INPUT:
    %   dataset - The name of dataset ('otb','vot2013','vot2014','vot2015')
    %   seqName - The name of a sequence in the given dataset
    %
    % OUTPUT:
    %   config - The configuration of the given sequence
    %
    % Hyeonseob Nam, 2015
    % 
% Input:
%       base_path - the base path of the sequences
%       seq_name -  the name of the sequence
% Output:
%       seq_info - a two-filed block including the path of each image file
%       (seq_info.img) and the groundtruth (seq_info.gt).

% By Xin Li, 2017-3-8
%% 
seq_info.seq_name=seq_name;

%% load images
% handle special sequences
switch(seq_info.seq_name)
    case {'Jogging-1', 'Jogging-2'}
        seq_info.img_dir = fullfile(base_path, 'Jogging', 'img');
    case {'Skating2-1', 'Skating2-2'}
        seq_info.img_dir = fullfile(base_path, 'Skating2', 'img');
    otherwise
        seq_info.img_dir = fullfile(base_path, seq_info.seq_name, 'img');
end

 if(~exist(seq_info.img_dir,'dir'))
        error('%s does not exist!!',seq_info.img_dir);
 end
        
 % parse img list
  seq_info.img_list = parseImg(seq_info.img_dir);
  switch(seq_info.seq_name)
      case 'David'
          seq_info.img_list = seq_info.img_list(300:end);
      case 'Tiger1'
          seq_info.img_list = seq_info.img_list(6:end);
  end
        
  % load gt
  switch(seq_info.seq_name)
      case 'Jogging-1'
          gtPath = fullfile(base_path, 'Jogging', 'groundtruth_rect.1.txt');
      case 'Jogging-2'
          gtPath = fullfile(base_path, 'Jogging', 'groundtruth_rect.2.txt');
      case 'Skating2-1'
          gtPath = fullfile(base_path, 'Skating2', 'groundtruth_rect.1.txt');
      case 'Skating2-2'
          gtPath = fullfile(base_path, 'Skating2', 'groundtruth_rect.2.txt');
      case 'Human4'
          gtPath = fullfile(base_path, 'Human4', 'groundtruth_rect.txt');
      otherwise
          gtPath = fullfile(base_path, seq_info.seq_name, 'groundtruth_rect.txt');
  end
        
  if(~exist(gtPath,'file'))
      error('%s does not exist!!',gtPath);
  end
        
  gt = importdata(gtPath);
  switch(seq_info.seq_name)
      case 'Tiger1'
          gt = gt(6:end,:);
      case {'Board','Twinnings'}
          gt = gt(1:end-1,:);
  end
        

  seq_info.gt = gt;
        
  nFrames = min(length(seq_info.img_list), size(seq_info.gt,1));
  seq_info.img_list = seq_info.img_list(1:nFrames);
  seq_info.gt = seq_info.gt(1:nFrames,:);
  seq_info.nFrames = nFrames;