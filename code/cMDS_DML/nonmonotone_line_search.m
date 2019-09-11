% made by yupanpan
% The code is designed for computing the Armijo line search


function[sigma]=nonmonotone_line_search(nei,L,X_train,eta,alpha,Y,c,K1)
%sigma=1*10^(-9);
sigma = 10;
c1 = 0.05;
rho = 0.5;
mk=0;
max_mk=20;
while mk<=max_mk
descent=devJ(L,X_train,eta,alpha,Y,c);
L1=L-sigma*descent;
f1=f(X_train,L1,c,Y,nei,alpha,K1);

f2=f(X_train,L,c,Y,nei,alpha,K1);
first = descent.^2;
second=sum(sum(first));
if f1> f2-c1*sigma*second
    sigma = rho*sigma;
else
    break
end
mk=mk+1;
    
end
