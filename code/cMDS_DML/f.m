function[final_function] = f(X_train,L,c,Y,nei,alpha,K1)
[d_train, n_train]  = size(X_train);
sss=L*X_train-c*Y;
sss=sss*sss';
jieguoone=trace(sss)*0.5;
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
jieguotwo=alpha*sum(sum(cha));
final_function=jieguoone+jieguotwo;
end