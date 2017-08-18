function [rind,w] = sweigthfrank(X)
%self weigth-based feature ranking
%reference: A novel intelligent method for bearing
%fault diagnosis based on affinity propagation clustering
%X in R^{N x P} : input matrix N samples P features

[N,P]  =size(X);
w = zeros(P,1);
for j = 1 : P
   w(j) = mean(pdist(X(:,j)).^2);
end
w = w - min(w);
w = w/max(w);
[~,rind] = sort(w,'descend');
