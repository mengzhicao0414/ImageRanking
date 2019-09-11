% made by yupanpan
% This code is the main to compare the method LDMLR and cMDS-DML
% The data is coming from UMIST dataset
% we randomly select K1 data from each distinct label as training data 
% we pick up subjects with age 1,5,9,15,19 years and relabel them as 1,2,3,4,5
clear
X=csvread('UMISTdata.csv');
r=csvread('UMISTlabel.csv');
num_repeat=50; 
T1=30;
T2=1000;
sigma1=1*10^(-20);
alpha1=10^(3);
alpha2=1*10^(-10);
K1=10; 
mySeed=(1:num_repeat)+20;

for i=1:num_repeat
    rng(mySeed(i));
    [n,d] =size(X);
    % construct training data and testing data 
    r_new = zeros(n,1);% generate n*1 0
    idx = find(r==0);
    idx_sample1=randsample(idx,K1);% generate sample index
    r_new(idx) = 0; %relabel

    idx = find(r==1);
    idx_sample2=randsample(idx,K1);
    r_new(idx) = 1;
   
    idx = find(r==2);
    idx_sample3=randsample(idx,K1);
    r_new(idx) = 2;
    
    % get X_train and its label r_train; get X_test and its label r_test
    idx_train=[idx_sample1;idx_sample2;idx_sample3];
    idx_test = setdiff([1:n]',idx_train);%idx_train, idx_test,pause  shan diao idx_train

    X_train = X(idx_train,:)'; X_test = X(idx_test,:)';
    r_train = r_new(idx_train); r_test = r_new(idx_test);
    K2=5; % the number of k-nearest neighbor 
    [MAE1,num_correct_full1,num_correct_round1,time1]=LDMLR(X_train,r_train,X_test,r_test,K2,sigma1,alpha1,T1); %LDMLR
    myTemp1.MAE(i) = MAE1;
    myTemp1.num_full(i) = num_correct_full1;
    myTemp1.num_round(i) = num_correct_round1;
    myTemp1.time(i)=time1;
    myResult1=myTemp1;
    [MAE2,num_correct_full2,num_correct_round2,time2]=cMDS_DML(X_train,r_train,X_test,r_test,K2,alpha2,T2); %cMDS-DML
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
