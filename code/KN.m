function[B]=KN(X_test0,X_train,D)

[d_train,n_train]=size(X_train);
F=zeros(n_train,1);
for i=1:n_train
    
        F(i,1)=(norm(X_test0-X_train(:,i)))^2;
    
end
e=eye(n_train,1);
B=-0.5*(F-(1/n_train)*D*e);
end