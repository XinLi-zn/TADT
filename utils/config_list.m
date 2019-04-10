function config = config_list( test_seq, test_source )
%Test file configuration
    if nargin <2
       test_source='../';     
    end
    imgList=parseImg([test_source '/' test_seq]);

    switch(test_seq)
        case 'David'
            imgList = imgList(300:end);
        case 'Tiger1'
            imgList = imgList(6:end);           
        otherwise
            
    end
    
    gtPath = fullfile(test_source, test_seq, 'groundtruth_rect.txt');
    if(~exist(gtPath,'file'))
        error('%s does not exist!!',gtPath);
    end

    gt = importdata(gtPath);
    switch(test_seq)
        case 'Tiger1'
            gt = gt(6:end,:);
        case {'Board','Twinnings'}
            gt = gt(1:end-1,:);
    end
    
    nFrames = min(length(imgList), size(gt,1));
    
    config.imgList=imgList;
    config.gt=gt;
    config.nFrames=nFrames;
    config.name=test_seq;
end

