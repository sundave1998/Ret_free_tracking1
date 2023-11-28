function Y = retraction_polar(X, U, t)
if nargin < 3
    Y = X + U;
else
    Y = X + t*U;
end

[u, ~, v] = svd(Y, 0); 
Y = u*v';
end