% made by yupanpan
% This code is designed for training L using cMDS-DML
% Step 1: search K1 target neighbours for each training data
% Step 2: using traditional stepest descent method with Armijo line search, solve the model
% min_{L,c} f(L,c)=1/2 sum_{eta_{ij}=1} (||Lx_i-cy_i||^2 + alpha sum(d^L(x_i,x_j)^2-d_I(x_i,x_j)^2)^2

% input:
% X_train          the training data
% r_train          the label of training data
% Y                the embedding data of training data
% alpha            a tradeoff parameter
% T                the iteration number

% output:
% L                the distance metric
function[L]=KNN(X_train,r_train,K1,Y,alpha,T)
r=r_train;
[d_train, n_train]  = size(X_train);
Delta1 = zeros(n_train,n_train);
nn_train=n_train-1;
for i = 1: nn_train
    for j = i+1:n_train
         tmp = X_train (:,i)-X_train (:,j);
         Delta1(i,j) =sqrt(tmp'*tmp);
    end
end
Delta1 = Delta1 + Delta1';
% Step 1
tmp=diag(inf+zeros(1,n_train));
Delta1=Delta1+tmp;
nei=zeros(n_train,K1);
for i=1:n_train
    for j=1:n_train
        if r(i)~=r(j)
            Delta1(i,j)=inf;
        end
    end
    temp=Delta1(i,:);
    [temp,index]=sort(temp);
    nei(i,:)=index(:,1:K1);   
end
eta=zeros(n_train,n_train);
for i=1:n_train
  eta(i,nei(i,:))=1;  
end
[d_Y,n_Y]=size(Y);
% Step 2
L=eye(d_Y,d_train);
c=zeros(T,1);
c(1)=1;
flag=0;
t=1;
while (t<T& flag==0)
    %c=1;   
    he1=0;
    he2=0;  
    for i=1:n_train
        he1=he1+Y(:,i)'*L*X_train(:,i);
        he2=he2+Y(:,i)'*Y(:,i);
    end
    c(t+1)=he1/he2;
    sigma=Armijo(nei,L,X_train,eta,alpha,Y,c(t),K1); % Armijo line search
  
    derivative=devJ(L,X_train,eta,alpha,Y,c(t)); % the gradient of f(L,c)
    L=L-sigma*derivative;
    derivative1=derivative.^2;
    sum_derivative=sum(sum(derivative1));
 
    sss=L*X_train-c(t+1)*Y;
    sss=sss*sss';
    jieguoone(t)=trace(sss)*0.5;
    dx=[];
    dx1=[];
    for j=1:n_train
    for i=1:K1
        XX=X_train(:,j)-X_train(:,nei(j,i));
        dx(j,i)=XX'*L'*L*XX;
        dx1(j,i)=XX'*XX;
        cha1(j,i)=dx(j,i)-dx1(j,i);
        cha(j,i)=(dx(j,i)-dx1(j,i))^2;
    end
    end
    jieguotwo(t)=alpha*sum(sum(cha));
    jieguothree(t)=jieguoone(t)+jieguotwo(t);
    if t>1 & (((sum_derivative/jieguothree(t))<1*10^(-3))|((abs(jieguothree(t)-jieguothree(t-1))/jieguothree(t-1))<5*10^(-4)))
        flag=1;
    end
    t=t+1;
end
%Z=zeros(n_train,n_train);
%for i=1:n_train-1
    %for j=i+1:n_train
       %Z(i,j)=(X_train(:,i)-X_train(:,j))'*L'*L*(X_train(:,i)-X_train(:,j)); 
    %end
%end
%Z=Z'+Z;
%plot( 1:n_train, Z(1,:),'.');
%xlabel('training data/x_i');
%ylabel('distance/D_1i');
%plot( 1:n_train, Z(1,:));

%LL=L*X_train;
%plot(jieguoone)
%plot(jieguotwo)
%plot(jieguothree)
end
