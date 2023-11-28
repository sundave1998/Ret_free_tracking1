function [A] = ring_graph(I)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code generates an undirected graph.
% 
% I: Number of agents over the graph
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
%%%% generate MH weight matrix %%%%
A = zeros(I);
if I == 1
    A = [1];
    
else
    if I == 2
      A = [0.5, 0.5; 0.5, 0.5];
    else
        
        for i = 1:I
            if i > 1 && i <I
                for j=i-1:i+1        
                    A(i,j) = 1 / 3;
                end
            end
        end
        A(1,1:2) =  1/3;
        A(1, I) = 1/3;
        A(I,1) = 1/3;
        A(I,I-1:I) = 1/3;
    end
end
 