% made by yupanpan
% This code is the main to compare the method LDMLR and cMDS-DML
% The data is coming from FG_NET dataset
% we randomly select K1 data from each distinct label as training data 

clear
X=csvread('FGNETdata.csv');
r=csvread('FGNETlabel.csv');
[n,d] =size(X);
% relabel the FG_NET dataset
idx1=find(r==1);
X1=X(idx1,:);
r1=r(idx1);

idx2=find(r==5);
X2=X(idx2,:);
r2=r(idx2);

idx3=find(r==9);
X3=X(idx3,:);
r3=r(idx3);

idx4=find(r==15);
X4=X(idx4,:);
r4=r(idx4);

idx5=find(r==19);
X5=X(idx5,:);
r5=r(idx5);

X_final=[X1;X2;X3;X4;X5];
r_final=[r1;r2;r3;r4;r5];
q=0;
for c=4:1:6
   
   
    q=q+1;
    q

n=length(r_final);
num_repeat=100;
T1=200;
T2=500;
sigma1=1*10^(-2);
alpha1=10^(3);
alpha2=10^(-10);
K1=10;
mySeed=(1:num_repeat)+60;


% for b=0.7:0.01:0.72
%     p=p+1;
%     q=0;
%     p
% for c=0:0.02:0.5
%    
%     q=q+1;
%     q
for i=1:num_repeat
    
    rng(mySeed(i));
    % construct training data and testing data
    r_new = zeros(n,1);% generate n*1 0
    idx = find(r_final==1);
    idx_sample1=randsample(idx,K1);% generate sample index
    r_new(idx) = 1; %relabel
   
    idx = find(r_final==5);
    idx_sample2=randsample(idx,K1);
    r_new(idx) = 2;
    
    idx = find(r_final==9);
    idx_sample3=randsample(idx,K1);
    r_new(idx) = 3;

    idx = find(r_final==15);
    idx_sample4=randsample(idx,K1);
    r_new(idx) = 4;
  
    idx = find(r_final==19);
    idx_sample5=randsample(idx,K1);
    r_new(idx) = 5;
    

    % get X_train and its label r_train; get X_test and its label r_test

    idx_train=[idx_sample1;idx_sample2;idx_sample3;idx_sample4;idx_sample5];
    idx_test = setdiff([1:n]',idx_train);%idx_train, idx_test,pause  shan diao idx_train
    %length(idx_test),pause
    X_train = X_final(idx_train,:)'; X_test = X_final(idx_test,:)';
    r_train = r_new(idx_train); r_test = r_new(idx_test);

    K2=c; % the number of k-nearest neighbor
%     [MAE1,num_correct_full1,num_correct_round1,time1]=LDMLR(X_train,r_train,X_test,r_test,K2,sigma1,alpha1,T1); %LDMLR
%     myTemp1.MAE(i) = MAE1;
%     myTemp1.num_full(i) = num_correct_full1;
%     myTemp1.num_round(i) = num_correct_round1;
%     myTemp1.time(i)=time1;
%     myResult1=myTemp1;

[MAE1,num_correct_full1,num_correct_round1,time1]=cMDS_DML(X_train,r_train,X_test,r_test,K2,alpha2,T2);  %cMDS-DML
    myTemp1.MAE(i) = MAE1;
    myTemp1.num_full(i) = num_correct_full1;
    myTemp1.num_round(i) = num_correct_round1;
    myTemp1.time(i)=time1;
    myResult1=myTemp1;
    [MAE2,num_correct_full2,num_correct_round2,time2]=norm_cMDS_DML2(X_train,r_train,X_test,r_test,K2,alpha2,T2,c);  %norm_cMDS-DML
     [MAE2,num_correct_full2,num_correct_round2,time2]=eMDS_DML2(X_train,r_train,X_test,r_test,K2,alpha2,T2,c); 
    myTemp2.MAE(i) = MAE2;
    myTemp2.num_full(i) = num_correct_full2;
    myTemp2.num_round(i) = num_correct_round2;
    myTemp2.time(i)=time2;
    myResult2=myTemp2;
    
end    
MyResult1.mean_MAE=mean(myResult1.MAE);
MyResult1.std=std(myResult1.MAE,0);
MyResult1.mean_num_full=mean(myResult1.num_full);
MyResult1.mean_num_round=mean(myResult1.num_round);
MyResult1.mean_time=mean(myResult1.time);

MyResult2.mean_MAE=mean(myResult2.MAE);
MyResult2.std=std(myResult2.MAE,0);
MyResult2.mean_num_full=mean(myResult2.num_full);
MyResult2.mean_num_round=mean(myResult2.num_round);
MyResult2.mean_time=mean(myResult2.time);

mytest2.MAE(q)=MyResult2.mean_MAE;% record different k_nearest neighbor
mytest2.std(q)=MyResult2.std;
mytest2.time(q)=MyResult2.mean_time;
end



