function [rind,w,A] = fdafrank(X,labels)
%pca-based feature ranking
%X in R^{N x P} : input matrix N samples P features
d = numel(unique(labels));
[~,A] = FDA(X',labels,d); %fda based projection
w = sum(abs(A),2);
w = w - min(w);
w = w/max(w);
[~,rind] = sort(w,'descend');
