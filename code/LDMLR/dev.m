% made by yupanpan
% The code is designed for computing the gradient of f(A):
% -sum (Omega_{ij} X_{ij}) + 2*alpha sum_{eta_{ij}=1}(X_{ij}(A-I)X_{ij})
% where f(A) = -sum(Omega_{ij} d_A(x_i,x_j)^2) + alpha sum(epsilon_{ij})
% (d_A(x_i,x_j)^2 -d_I(x_i,x_j)^2)^2 = epsilon_{ij}, if eta_{ij}=1， A>=0, epsilon_{ij}>=0 
% input:
% A               the distance metric
% X_train         training data 
% Omega           the weighting factor
% eta             target neighbors matrix
% alpha           the tradeoff parameter

% output:
% zuihou          the gradient
function[zuihou]=dev(A,X_train,Omega,eta,alpha)
[d_train,n_train]=size(X_train);
zuihou=zeros(d_train,d_train);
diag_I=eye(d_train,d_train);
    for i=1:n_train
        for j=1:n_train
            zuihou=zuihou-Omega(i,j)*((X_train(:,i)-X_train(:,j))*(X_train(:,i)-X_train(:,j))'); 
        end
    end
    for i=1:n_train
        for j=1:n_train
            if eta(i,j)==1
                ccc=X_train(:,i)-X_train(:,j);
                first=ccc'*A*ccc;
                second=ccc*first*ccc';
                third=ccc'*ccc;
                forth=ccc*third*ccc';
                five=second-forth;
                zuihou=zuihou+2*alpha*five;
            end
        end
    end
end
