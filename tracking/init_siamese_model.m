function [net_matching] = init_siamese_model()
% this file implements the matching network
% By Xin Li, 11-27-2017

net_matching = dagnn.DagNN();
    net_matching.addLayer('xcorr', XCorr(), ...
                 {'w_feat', 'x_feat'}, ...
                 {'m_score'}, ...
                 {});
net_matching.move('gpu');

end