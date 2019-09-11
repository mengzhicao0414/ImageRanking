% made by yupanpan
% This code is the main procedure of cMDS-MDL
% Step 1: applying cMDS
% Step 2: training L
% Step 3: testing
% the main part of code is in testing procedure
% use k-nearest neighbor regression to obtain the estimated label hat(r_i) 
% measure the performance:
% MAE = 1/N sum(hat(r_i)-r_i), where r_i is the true label

% input:
% X_train     training data
% r_train     the label of training data
% X_test      testing data
% r_test      the true label of training data
% K2          the parameter of k-nearest neighbor 
% alpha       a parameter in KNN1
% T           the max iteration number

% output:
% MAE
% num_correct_full            the total number of hat(r_i)==r_i
% num_correct_round           the total number of round(hat(r_i))==r_i
% time                        training time


function[MAE,num_correct_full,num_correct_round,time]=cMDS_DML(X_train,r_train,X_test,r_test,K2,alpha,T)
t0=cputime;
% Step 1
[Y]=MDS(X_train,r_train);
Y
% Step 2
[L]=KNN(X_train,r_train,K2,Y,alpha,T);
time=cputime-t0;
[d_train,n_train]=size(X_train);
n_test = length(r_test);
r_hat = zeros(n_test,1);
% Step 3
Classlabel=unique(r_train);
Lengthc=length(Classlabel);

for i = 1: n_test
    cnt=zeros(Lengthc,1);
    x = X_test(:,i);
    dx=zeros(n_train,1);
   
    tmp = repmat(x',n_train,1);
    tmp = L*(tmp'-X_train);
    dx= sum(tmp.*tmp,1);
   
    [dx_new, idx] = sort(dx,'ascend');
    for k=1:K2
        ind=find(Classlabel==r_train(idx(k)));
        cnt(ind)=cnt(ind)+1;
    end
    [m,ind]=max(cnt);
    r_hat(i)=Classlabel(ind);
end

% measure the performance
MAE= mean(abs(r_hat-r_test));
r_estimate_round = round(r_hat);
num_correct_round = length(find((r_estimate_round-r_test)==0));
num_correct_full = length(find((r_hat - r_test)==0));
end
