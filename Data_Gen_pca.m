function [A] = Data_Gen_pca(Num_Nodes, n, m)
 
A = cell(Num_Nodes,1);
for i = 1:Num_Nodes
    A{i} = randn(n ,m);
    
end
end