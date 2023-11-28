function px = proj_tangent(x,d)
xd = x'*d;
px = d - 0.5*x*(xd+xd');