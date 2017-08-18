function [Xn,maxX,minX] = drnormalization(X)
%dynamic range data normalization
maxX = max(max(X));
minX = min(min(X));
Xn = (X - minX)./(maxX-minX);
