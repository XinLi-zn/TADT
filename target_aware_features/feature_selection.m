% this filter selects features from the input feature feat based on the
% specified method.
% Input:
%   feat    -   features to be selected, matrix
%   coef    -   coefficients used in the selection
%   method  -   feature selection method. {'pca', 'sa'}
% Output:
%   sel_feat-   selected features

function [sel_feat] = feature_selection(feat, coef, method, layer_w)


switch method    
    case 'pca'
        [ sel_feat] = do_pca( feat, coef );
    case 'sa'
        coef = coef{1};
%         sel_feat{1} = bsxfun(@times, feat{1}(:,:,coef>0,:), reshape(coef(coef>0),1,1,[]));
        sel_feat{1} = bsxfun(@times, feat{1}, coef);
    case 'pca_sa'
        assert(numel(coef)==2,'coefficients and feature selection method does not compatient!');
        [ sel_feat] = do_pca( feat, coef(1));
        sel_feat{1} = sel_feat{1}(:,:,coef{2}==1,:);
    case 'Wreduction'
        num = numel(feat);
        assert(numel(coef)==num,'weights number and feature number are not identical!\n');
        patch_sw = cellfun(@(fea,coe)  fea(:,:,coe>0),...
                                feat,coef,'uniformoutput', false);
                           
        sel_feat = {cat(3,patch_sw{1},patch_sw{2}*layer_w(2))};
    otherwise
        
end

end