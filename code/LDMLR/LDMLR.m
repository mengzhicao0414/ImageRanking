% made by yupanpan
% The code is designed for LDMLR
% min_A f(A) = -sum(Omega_{ij} d_A(x_i,x_j)^2) + alpha sum(epsilon_{ij})
% s.t. (d_A(x_i,x_j)^2 -d_I(x_i,x_j)^2)^2 = epsilon_{ij}
%      if eta_{ij}=1， A>=0, epsilon_{ij}>=0 
% eta_{ij} =1, if x_j is one of x_i's target neighbors
% eta_{ij} =0, otherwise
% Omega_{ij} = sqrt(|r_i-r_j|+1), if r_i is not equal to r_j
% Omega_{ij} =0, otherwise

% employ a gradient descend method to solve the model

% Step 1: search K2 target neighbors for each data point
% Step 2: iterative to get A, including compute the gradient of f(A) and make A positive semidefinite
% for the testing, similar with myarticle.m

% input:
% X_train     training data
% r_train     the label of training data
% X_test      testing data
% r_test      the true label of training data
% K2          the parameter of k-nearest neighbor
% sigma       the learning rate 
% alpha       the tradeoff parameter
% T           the max iteration number

% output:
% MAE
% num_correct_full            the total number of hat(r_i)==r_i
% num_correct_round           the total number of round(hat(r_i))==r_i
% time                        training time

function[MAE,num_correct_full,num_correct_round,time]=LDMLR(X_train,r_train,X_test,r_test,K2,sigma,alpha,T)
r=r_train;
[d_train, n_train]  = size(X_train);
% Step 1
Delta = zeros(n_train,n_train);
nn_train=n_train-1;
for i = 1: nn_train
    for j = i+1:n_train
         tmp = X_train (:,i)-X_train (:,j);
         Delta(i,j) =sqrt(tmp'*tmp);
    end
end
Delta = Delta + Delta';
%subplot(1,2,1)
%plot( 1:n_train, Delta(1,:));
tmp=diag(inf+zeros(1,n_train));
Delta=Delta+tmp;
nei=zeros(n_train,K2);
for i=1:n_train
    for j=1:n_train
        if r(i)~=r(j)
            Delta(i,j)=inf;
        end
    end
    temp=Delta(i,:);
    [temp,index]=sort(temp);
    nei(i,:)=index(:,1:K2);   
end
eta=zeros(n_train,n_train);
for i=1:n_train
  eta(i,nei(i,:))=1;  
end
% Step 2
Omega=zeros(n_train,n_train);
p=0.5;
for i=1:n_train-1
    for j=i+1:n_train
        if r(i)~=r(j)
            Omega(i,j)=(abs(r(i)-r(j))+1)^p;
        end
    end
end
Omega=Omega+Omega';
%sigma=1;
t0=cputime;
%sigma=1*10^(-7);
%alpha=10^3;
A=zeros(d_train,d_train);
%T=50;
 t=1; 
 flag=0;
while t<T  & flag==0
    A=A-sigma*dev(A,X_train,Omega,eta,alpha);
    A=PSD(A,d_train);
    dx=[];
    dx1=[];
    for j=1:n_train
    for i=1:K2
        XX=X_train(:,j)-X_train(:,nei(j,i));
        dx(j,i)=XX'*A*XX;
        dx1(j,i)=XX'*XX;
        cha(j,i)=(dx(j,i)-dx1(j,i))^2;
    end
    end
    jieguotwo1(t)=sum(sum(cha));
    for j=1:n_train
    for i=1:n_train
        XX=X_train(:,j)-X_train(:,i);
        dx(j,i)=Omega(j,i)*XX'*A*XX;
    end
    end
    jieguotwo2(t)=-sum(sum(dx));
    jieguotwo3(t)=jieguotwo2(t)+alpha*jieguotwo1(t);
    if t>1& (abs((jieguotwo3(t)-jieguotwo3(t-1))/jieguotwo3(t-1)))<1*10^(-3)
       flag=1;
   end
   t=t+1;
    %plot(cha);
end
%plot(jieguotwo2)
%plot(jieguotwo3)
time=cputime-t0;

A=A+A';
A=A*0.5;
[temp1,temp2]=eig(A);
temp3=size(temp2);
for cc=1:temp3(1)
    if(temp2(cc,cc)<0)
        temp2(cc,cc)=0;
    end
end
temp4=sqrt(temp2);

D=zeros(n_train,n_train);
    for i=1:n_train-1
        for j=i+1:n_train
            tmp = X_train (:,i)-X_train (:,j);
            xxxx=temp4*temp1'*tmp;
            D(i,j) =sqrt(xxxx'*xxxx);
        end
    end
D=D+D';
%subplot(1,2,2)
%plot( 1:n_train, D(1,:));
    
% the testing part   
n_test = length(r_test);
r_hat = zeros(n_test,1);
Classlabel=unique(r_train);
Lengthc=length(Classlabel);

for i = 1: n_test
    cnt=zeros(Lengthc,1);
    x = X_test(:,i);
    dx=zeros(n_train,1);
    for j=1:n_train
        tmp=x-X_train(:,j);
        dx(j,:)=tmp'*A*tmp;
    end
   
    [dx_new, idx] = sort(dx,'ascend');
    for k=1:K2
        ind=find(Classlabel==r_train(idx(k)));
        cnt(ind)=cnt(ind)+1;
    end
    [m,ind]=max(cnt);
    r_hat(i)=Classlabel(ind);
    

end
r_estimate_round = round(r_hat);
MAE = mean(abs(r_hat-r_test));

% for i = 1: n_test
%     x = X_test(:,i);
%     dx=zeros(n_train,1);
%     for j=1:n_train
%         tmp=x-X_train(:,j);
%         dx(j,:)=tmp'*A*tmp;
%     end
%    
%     [dx_new, idx] = sort(dx,'ascend');
%     
%     lambda = 1./dx_new(1:K2);
%     if sum(lambda)==inf
%       r_hat(i) = r_test(idx(1));  
%     else
%     r_i = r_train(idx(1:K2));
%     r_hat(i) = sum(r_i.*lambda)/sum(lambda);
%     end
% end
% r_estimate_round = round(r_hat);
% MAE = mean(abs(r_hat-r_test));

%r_estimate_ceil = ceil(r_hat);
%r_estimate_floor = floor(r_hat);
%num_correct_ceil = length(find((r_estimate_ceil-r_test)==0)) ; 
%num_correct_floor = length(find((r_estimate_floor-r_test)==0)); 
num_correct_round = length(find((r_estimate_round-r_test)==0));
num_correct_full = length(find((r_hat - r_test)==0));
r_estimate_round=round(r_hat);
num_correct_round=length(find((r_estimate_round-r_test)==0));
end


