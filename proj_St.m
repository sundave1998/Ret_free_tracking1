function [px, nuclear_norm] = proj_St(x,N)
% [d, r] = size(x);
[u,s,v] = svd(x,0);
px = u*v';
nuclear_norm = sum(sum(s))/N;
