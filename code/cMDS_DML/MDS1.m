% made by yupanpan
% This code is designed for applying cMDS, to get embedding data pf the training data
% Step 1: construct an EDM based on the label r_1, r_2, ..., r_n of training data 
% D_ij = (|r_i-r_j|+ 1)^2, if r_i is not equal to r_j
% D_ij = 0, if r_i=r_j

% Step 2: compute the double-centralized matrix B of D
% J =  I-1/n 11^T, where I is n×n identity matrix and 1 is the vector of all ones in R^n.
% B = -1/2 JDJ 

% Step 3: get embedding data y_1, y_2, ..., y_n, and the embedding dimension is m=3
% the spectral decomposition of B is 
% B = [p_1, ..., p_s] diag(lambda_1, ..., lambda_s) [p_1, ..., p_s]^T
% y_1,y_2, ..., y_n = diag(sqrt(lambda_1), ..., sqrt(lambda_{K1})) [p_1, ..., p_{K1}]^T

% Input:
% X_train        training data
% r_train        the label of training data

% Output:  
% Y              embedding data of training data


function[Y1,Y,D]=MDS1(X_train,r_train,c)
[d, n_train]  = size(X_train);
% J=eye(n_train)-n_train^(-1)*ones(n_train)
% for i=1:n_train
%     for j=1:n_train
%         dio(i,j)=abs(r_train(i)-r_train(j));
%     end
% end
% 
% K=-0.5*J*dio*J
% u0=eig(K)
nn_train=n_train-1;
%designed D
Delta = zeros(n_train,n_train);
% max0=norm(X_train(:,1));
% for i=1: n_train
%     if max0<norm(X_train(:,i))
%         max0=norm(X_train(:,i));
%     end
% end
for i = 1: n_train-1                        
    for j = i+1:n_train
        if r_train(i)~=r_train(j)
           temp=norm (X_train(:,i)-X_train(:,j));
        Delta(i,j)=(sqrt(abs(r_train(i)-r_train(j))^2+5.5^2*temp^2)-0)^2; 
        end
    end
end
Delta = Delta + Delta';
D=Delta;
eigsolver=0;
prnt=0;
error_tol=1.0e-6;
lol=length(D);
vector=zeros(lol,1);
prnt=0;

[D,vector,infos] = ENewton(D, eigsolver, prnt, vector, error_tol);

% compute J and B
first=ones(n_train,1);
second=first'*D;
forth=second*first;
third=D-n_train^(-1)*first*second-n_train^(-1)*D*first*first'+n_train^(-2)*first*forth*first';
B = -third*0.5;
B = (B + B')*0.5;
% spectral decomposition of B
[P,lambda0] = eig(B);
lambda0 = diag(lambda0);
[lambda0,idx] = sort(lambda0,'descend');
P = P(:,idx);
m=3; % the embedding dimension
% get Y
P1=P(:,1:m);
Y = diag(sqrt(lambda0(1:m).^(-1)))*P1';
Y1 = P1* diag(sqrt(lambda0(1:m)));
Y1=Y1';
% Y=Y';
% plot the square distance between y_1 and y_i, i.e. ||y_1-y_i||^2
% Z=zeros(n_train,n_train);
% for i=1:n_train-1
%     for j=i+1:n_train
%        Z(i,j)=(Y(:,i)-Y(:,j))'*(Y(:,i)-Y(:,j)); 
%     end
% end
% Z=Z'+Z;
% plot( 1:n_train, Z(1,:));
end
