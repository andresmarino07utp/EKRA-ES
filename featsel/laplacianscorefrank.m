function [rind,w,K,sigma] = laplacianscorefrank(X,labels,k)
%laplacian score using a gaussian kernel
%reference: Laplacian score for feature selection. Xiaofei, He.
%X in R^{N x P} : input matrix N samples P features
%k in N : number of nearest neighbors for building the graph
[N,P]  =size(X);
w = zeros(P,1);

if nargin < 2
     k = round(sqrt(N));   
     labels = ones(size(X,1),1);
elseif nargin < 3
    k = round(sqrt(N));
end
%build the graph
D = pdist2(X,X);
[~,ind] = sort(D,'ascend');
sigma = 0.5*median(median(D(ind(2:k+1,:))));
K = exp(-D.^2/(2*sigma^2));
for i = 1 : N
    K(ind(k+1:end,i),i) = 0; 
end

L = bsxfun(@eq,labels,labels');
K = K.*L;


%build the degree matrix
Ds = diag(sum(K,2));
%build the graph laplacian
L = Ds - K;
%normalized feature value
Xn = X - repmat((X'*Ds*ones(N,1)/(ones(1,N)*D*ones(N,1)))',N,1);
for j = 1 : P
    w(j) = Xn(:,j)'*L*Xn(:,j)./(Xn(:,j)'*D*Xn(:,j));
end
w = w - min(w);
w = w/max(w);
[~,rind] = sort(w,'descend');
