% made by yupanpan
% The code is designed for computing the gradient of f(L,c)
% the gradient f(L,c) on L:
% = sum(Lx_i x_i^T-cy_i c_i^T) + 4 alpha sum (||L(x_i-x_j)||^2-||x_i-x_j||^2) L(x_i -x_j)(x_i-x_j)^T

function[Lfinal]=devJ(L,X_train,eta,alpha,Y,c)
[d_train,n_train]=size(X_train);
[d_Y,n_Y]=size(Y);

Lfinal=zeros(d_Y,d_train);
%daoshu1=zeros(d_Y,d_train);
%daoshu2=zeros(d_Y,d_train);
%tempA=L'*L;
diag_I=eye(d_train,d_train);
    for i=1:n_train
            Lfinal=Lfinal+(L*X_train(:,i)*X_train(:,i)'-c*Y(:,i)*X_train(:,i)'); 
    end
    for i=1:n_train
        for j=1:n_train
            if eta(i,j)==1
                ccc=X_train(:,i)-X_train(:,j);
                first=L*ccc;
                second=first'*first;
                third=first*second*ccc'; 
                forth=ccc'*ccc;
                five=first*forth*ccc';
                six=third-five;
                Lfinal=Lfinal+4*alpha*six;
            end
        end
    end
end
