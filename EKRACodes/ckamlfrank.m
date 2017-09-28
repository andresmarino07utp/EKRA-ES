function [rind,w,A,L,opts] = ckamlfrank(X,labels,A_i)
%cka-based metric learning for feature ranking
%X in R^{N x P} : input matrix N samples P features
%labels in N^{N} : label vector
% Andres Marino Alvarez Meza, Automatics Research Group
% Universidad Tecnologica de Pereira, Pereira - Colombia
% email: andres.alvarez1@utp.edu.co
[n,nl] = size(labels);
if nl > n
    labels = labels';
end

if nargin < 3
    opts.init = 'pca';
else
     opts.init = A_i;
end


L = bsxfun(@eq,labels,labels'); %label kernel.* amplitude-pulse kernel
[~,ind] = sort(labels);
opts.maxiter = 250;
opts.showWindow = false;
opts.showCommandLine = true;
[~, A_i]=ml_pca(X,0.95);
opts.Q = size(A_i,2);
if size(A_i,2) < 3
   opts.Q = 3;
end   
A = kMetricLearningMahalanobis(X(ind,:),L(ind,ind),labels(ind),opts);
w = sum(abs(A),2);
w = w - min(w);
w = w/max(w);
[~,rind] = sort(w,'descend');


