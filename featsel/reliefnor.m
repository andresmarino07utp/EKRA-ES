function [rind,w] = reliefnor(X,labels,kn)
% relieff feature selection using 1-KNN prediction
%X in R^{N x P} : input matrix N samples P features
%labels in N^N : label vector

if nargin < 3
    kn = 1;
end
[~,w]=relieff(X,labels,kn);
w = w - min(w);
w = w/max(w);
w = w';

[~,rind] = sort(w,'descend');

