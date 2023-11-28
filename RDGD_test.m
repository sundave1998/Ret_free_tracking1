clear;

%%%% Problem parameters %%%%
N                 = 4;
m                 = 1000;    % number of samples per node
n                 = 100;     % problem dimension
r                 = 5;     % column number
maxiter           = 10000;
Num_trials        = 1;

%%%% Initializations %%%%

for iter = 1:Num_trials
    
    fprintf('trial: %d\n',iter);
    %     rng(10)
    x0 = randn( n, r);
    [x0,~] = qr(x0,0);
    for i = 1:N
%         [u,~,v] = svd(x0 + 0*randn(n,r),0);
        X0{i} = x0;
    end
    A = Data_Gen_pca(N, n, m);
    B = zeros(m*N, n);
    for i = 1:N
        B(m*(i-1)+1:m*i,:) = A{i}';
    end
    for i = 1:1
        [u,s,v] = svds(B, n);
        singular_val = diag(s);
        for j = 2:n
            singular_val(j) = singular_val(1)*0.8^(j-1);
        end
        B = u*diag(singular_val)*v';
    end
    for i =1:N
        A{i} = B(m*(i-1)+1:m*i,:)';
    end
    
    A_m = zeros(n);
    for i =1:N
        A_m = A_m + A{i}*A{i}';
    end
    
    [xopt,~] = svds(A_m,r);
    
    fopt = sum(sum(xopt'*A_m*xopt));
    %     W = ER_graph(N, 0.3);
    W = ring_graph(N);
    
    Lam = svds(W,2);
%     mu_2 = Lam(2);
    
    %        [X,x_mean, distance, F_val]  = RDGD(A, W,stepsize, X0, maxiter);
    %      [X,x_mean, distance, F_val]  = GPM_decentralized(A, W,stepsize, X0, maxiter);
    %     [X ,x_mean, distance] = GPM_tracking(A, W,stepsize,X0, maxiter);
    % tic
    % stepsize =  0.1/m;
    % [X ,x_mean, c_err, distance] = RDGD_tracking(A, W,stepsize,1,X0, maxiter,xopt);
    % toc
    tic
    stepsize_0 = 0.1/m;
    [X_1 ,x_mean_1, c_err, distance_1] = Ret_free_tracking1(A, W,stepsize_0,1,X0, maxiter,xopt);
    [X_2 ,x_mean_2, c_err, distance_2] = Mod_Ret_free_tracking1(A, W,stepsize_0,1,X0, maxiter,xopt);
    toc
%     stepsize_1 =  0.1/m;
%     [X ,x_mean, c_err, distance_1] = Ret_free_tracking2(A, W,stepsize_1,1,X0, maxiter,xopt);
    %      stepsize =  0.2/m;
    %     [X_1 ,x_mean_1, c_err_1, distance_1] = RDGD_tracking(A, W,stepsize,5,X0, maxiter,xopt);
 
    
end

% x = x0;
% for i = 1:2000
%     grad = A_m*x;
%     pgd =  proj_tangent(x, grad);
%     rgrad(i) = norm(pgd, 'fro')^2;
%     f(i) = sum(sum(x.*grad));
%     theta = 1/n;
%     [U,~,V] = svd(x+ theta*grad,0);
%     x = U*V';
% end
% semilogy(rgrad,'g');
% hold on

% semilogy(distance, 'r');
% hold on

semilogy(distance_1, 'b-.');
hold on

semilogy(distance_2, 'g-.');
% hold on

% hold on;
% semilogy(distance_1, 'g');
legend( 'DRGTA', 'Ret-free', 'Ret-test', 'Location','best');
% legend( 't=1, small beta','DSA', 't=10, larger beta', 'Location','best');



