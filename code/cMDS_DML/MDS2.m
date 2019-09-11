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


function[Y]=MDS2(X_train,r_train,c)
[d_train, n_train]  = size(X_train);
nn_train=n_train-1;
%designed D
Delta = zeros(n_train,n_train);
% C=zeros(d_train,1);
% max0=zeros(d_train,1);
% 
% for u =1:d_train
%     
% for k = 1: n_train-1                        
%     for s = k+1:n_train
%         
%         
%             
%      tmp(k,s)= abs(r_train(k)-r_train(s))^2*(X_train(u,k)-X_train(u,s))^(-2);
%            if tmp(k,s)>max0(u)
%                max0(u)=tmp(k,s);
%            end
%         end
%     end
% end

% [max0,idx]=sort(max0,'descend');
% 
% C(idx(1))=max0(1)*0.000000000001;
% C


for i = 1: n_train-1                        
    for j = i+1:n_train
        if r_train(i)~=r_train(j)
            temp=0;
            
            for k=1:d_train
                temp=temp+(abs(X_train(k,i)-X_train(k,j)))^2;
            end
%             Delta(i,j)=(temp^(0.5)+randn)^2;
        Delta(i,j)=(sqrt(temp)+abs(r_train(i)-r_train(j)))^2;
%          Delta(i,j)=(3*(temp)^(1/1.5)+abs(r_train(i)-r_train(j))-0.5)^2;
%      Delta(i,j)=((max0'*(X_train(:,i)-X_train(:,j)))^2+abs(r_train(i)-r_train(j)))^2;
%   Delta(i,j)=((max0'*(X_train(:,i)-X_train(:,j)))^2+abs(r_train(i)-r_train(j)))^2
         
        end
    end
end
Delta = Delta + Delta';
D=Delta;
% eigsolver=0;
% prnt=0;
% error_tol=1.0e-6;
% lol=length(D);
% vector=zeros(lol,1);
% prnt=0;
% 
% [D,vector,infos] = ENewton(D, eigsolver, prnt, vector, error_tol);
for i = 1:n_train
    for j= 1:n_train
%         pars.H(i,j)=exp(0.5*abs(r_train(i)-r_train(j)));
 pars.H(i,j)=(abs(r_train(i)-r_train(j))+1)^1;
%  pars.H(i,j)=1;
    end
end
    pars.b=[];
    pars.I=[];
    pars.J=[];
    pars.eig=1;
    pars.r=20;
%      pars.Y0=[];
%     pars.tolrank=1.0e-8;
%     pars.tolrel=1.0e-5;
%     pars.spathyes=1;
pars.printyes=0;
    pars.plotvarianceyes=0;
     pars.plot2dimyes=0;
      pars.plot3dimyes=0;
 
D = sparse(D);
[D, infos] = rHENewton2_beta(D, pars);

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
Y = P1* diag(sqrt(lambda0(1:m)));
Y=Y';
% plot the square distance between y_1 and y_i, i.e. ||y_1-y_i||^2
Z=zeros(n_train,n_train);
for i=1:n_train-1
    for j=i+1:n_train
       Z(i,j)=(Y(:,i)-Y(:,j))'*(Y(:,i)-Y(:,j)); 
    end
end
Z=Z'+Z;
plot( 1:n_train, Z(1,:));
end
