function [X,avg_x,Err, dist] = Ret_free_tracking1(A, W,stepsize,multistep,X0, maxiter,xopt)
N = size(W,1);
[n, r] = size(X0{1});
Err    = zeros(1, maxiter); % to record the optimality gap at each round
f      = zeros(1, maxiter);
%%%% Initialization %%%%
lambda = 0.1;
X  = X0;
y = cell(N,1);
for i = 1:N
    xtx{i} = X{i}'*X{i};
    p{i} = lambda*X{i}*(xtx{i} - eye(r));
    y{i} =  A{i}*A{i}'*X{i};  % initial tracking
    y{i} = proj_tangent(X{i}, y{i});
end
Y0 = cell(N,1);
for i = 1:N
    Y0{i} = zeros(n,r);
end

grad = y;
for k = 1:maxiter
    %%%%  update %%%%
    c_old = X;
    g_old = grad;
    y_old  = y;
    y_0 = y;
    
    for T = 1:multistep
        c = Y0;
        y = Y0;
        for i = 1:N
            for j = 1:N
                if W(i,j) ~= 0 %&& j ~= i
                    c{i} =  c{i} +  W(i,j)*c_old{j};
                    y{i} = y{i} + W(i,j)*y_old{j};
                end
            end
        end
        c_old = c;
        y_old = y;
    end
    
    %     max_y = 0;
    for i = 1:N
        % new_d = proj_tangent(X{i}, 0.5*c{i} + stepsize*y{i});
        new_d = proj_tangent(X{i}, 0.5*c{i} + stepsize*grad{i});
        %         [U,~,V] = svd(X{i} + new_d,0);
        %         X{i} = U*V';
        X{i} = X{i} + new_d - p{i};
        xtx{i} = X{i}'*X{i};     
        p{i} = lambda*X{i}*( xtx{i} - eye(r));
        grad{i} =  A{i}*A{i}'*X{i};
        grad{i} = proj_tangent(X{i}, grad{i});
        y{i} = y{i} + grad{i} - g_old{i};
    end
    
    %     f(k)   = trace((X{i}'*A_sum)*(A_sum'*X{i}));
    avg_x = zeros(n,1);
    sum_g = zeros(n,r);
    max_y = 0;
    for i = 1:N
        avg_x = avg_x + X{i};
        sum_g = sum_g + y{i};
        max_y = max(max_y, norm(y{i},'fro'));
    end
    
    [avg_x, n_norm] = proj_St(avg_x, N);
    Err(k) = sqrt(2*N*abs(r-n_norm))/N;
    normg(k) = norm(sum_g,'fro')/N;
    [u,s,v] = svd(xopt'*avg_x);
    dist(k) = sqrt(2*abs(r-sum(sum(s))));
    if Err(k)  < 1e-8 && N ~= 1 && normg(k) < 1e-5
        break;
    end
    if dist(k) < 1e-7 && Err(k)  < 1e-8
        break;
    end
    if mod(k, 100) == 0
        fprintf('RDGD_tracking %d-th iter, consensus error: %1.3e  gradient:  %1.3e  maxy:  %1.3e\n', k, Err(k),normg(k), max_y);
        % fprintf('RDGD_tracking: %d-th round, the error of y is %1.3e\n', k, Feasiblity);
    end
    
end
end

