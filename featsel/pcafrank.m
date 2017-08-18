function [rind,w,A] = pcafrank(X)
%pca-based feature ranking
%X in R^{N x P} : input matrix N samples P features
d = 0.95;
[~,A,Val] = A_pca(X,d);
w = sum(abs(repmat(Val(1:size(A,2))',size(A,1),1).*A),2);
w = w - min(w);
w = w/max(w);
[~,rind] = sort(w,'descend');
