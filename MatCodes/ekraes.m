function [Y,rho,ind,As,Ap] = ekraes(X,labels)
% ekra-es relevance analysis
% Function basics
%Alvarez, Orozco & Castellanos  Kernel-based Relevance Analysis with Enhanced Interpretability for Detection of Brain Activity Patterns
%Brockmeier et. al. Neural Decoding with kernel-based metric learning
%Cardenas & Alvarez  Sigma tune Gaussian kernel with information potential

%USAGE:
% [Y,rho,ind] = ekraes(X,L,labels)
%INPUTS:
% X \in R^{N x P}      : Data matrix, N: samples; P:features
% labels \in Z^{N x 1} : Group membership
% OUTPUTS:
% Y \in R^{P x M_e}       : mapped feature matrix 
% rho \in P               : relevance vector
% ind \in P               : feature indexes sorted according to rho
% As \in R^{P x M_s}      : Learned rotation matrix by maximizing centered 
% Ap \in R^{P x M_e}      : Learned rotation matrix by maximizing centered 

% Andres Marino Alvarez Meza, Automatics Research Group
% Universidad Tecnologica de Pereira, Pereira - Colombia
% email: andres.alvarez1@utp.edu.co
%% EKRA - S Feature selection approach
fprintf('EKRA-S...\n')
tic 
[ind,rho,As,L,opts] = ckamlfrank(X,labels);
toc
thrho = mean(rho);
%% EKRA -ES -> Feature extraction from relevant subset
Xs = X(:,rho > thrho); %find relevant subset
fprintf('EKRA-ES...\n')
tic
Ap = kMetricLearningMahalanobis(Xs,L,labels,opts);
Y = Xs*Ap; %mapped feature matrix
toc
fprintf('done\n')