function [rind,w] = distsupfrank(X,labels)
%distance-based feature selection with class label information
%X in R^{N x P} : input matrix N samples P features
%labels in N^{N} : label vector
% referencia: Lukas Jedlinski, Jozef Jonak. A disassembly-free method for evaluation of spiral bevel gear
% assembly 2017
[n,nl] = size(labels);
if nl > n
    labels = labels';
end
[N,P] = size(X);
vc = unique(labels);
C = numel(vc);
Dc = zeros(C,P);
for c = 1 : C
    for j = 1 : P
        indc = labels == vc(c);
        Nc = sum(indc);
        %mean distance between individual features for the same condition
        Dc(c,j) = mean(pdist(X(indc,j)));
        %mean value for all features into the same condition
        ac(c,j) = mean(X(indc,j));
    end
end
%mean distance for C classes
Dwj = mean(Dc);
%compute variance coefficient
mind = min(Dc);
mind(abs(mind) < eps ) = 1e-10;
Vwj = max(Dc)./mind;
%mean distance between features for different conditions
for j = 1 : P
   Dbj(j) = mean(pdist(ac(:,j)));
   Dacj(:,j) = pdist(ac(:,j),'cityblock');
   minda = min(Dacj(:,j));
   if abs(minda) < eps
       minda = 1e-10;
   end 
   Vbj(j) = max(Dacj(:,j))/minda;
end
%lambda value
lj = (Vwj./max(Vwj) + Vbj./max(Vbj)).^(-1);
Ej = lj.*(Dbj./Dwj);
w = Ej/max(Ej);
w = w';
w = w - min(w);
w = w/max(w);
[~,rind] = sort(w,'descend');

