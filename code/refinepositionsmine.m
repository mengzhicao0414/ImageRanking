%%*************************************************************************
%% refinepositions: steepest descent with nonmonotone line search to 
%%                  minimize 
%%
%% f(X) = sum_{j<=k: djk is given}  (norm(xj-xk)-djk)^2 
%%        + sum_{j,k: djk is given} (norm(xj-ak)-djk)^2
%%
%% input: X0     = [sensor position in column format]
%%        anchor = [anchor position in column format] 
%%        DD     = [sensor-sensor distance, senor-anchor distance
%%                  anchor-sensor distance,  0]
%% (optional) maxit = maximum number of gradient iterations allowed 
%% (optional) tol   = stopping tolerance  
%%
%% output: Xiter = [refined sensor position]
%%         Info.objective  = history of objective values
%%         Info.gradnorm   = history of norm of the gradients
%%         Info.steplength = history of step-lengths
%%         Info.cputime    = cputime taken
%% 
%% child functions: gradfun2, objfun2
%%*************************************************************************


  function [Xiter,Info] = refinepositionsmine(X0,anchor,DD,maxit,no_first,tol)

  if ~exist('maxit'); maxit = 500; end
  if ~exist('tol'); tol = 1e-9; end%1e-9

  ttime = clock; 
 
  [dummy,n] = size(X0);
  [dummy,m] = size(anchor); 
  if any(size(DD) ~= [n,n+m]) 
     error('refinepositions: dimension of X0 or DD not correct')
  end
  D1 = DD(1:n,1:n); 
  [II,JJ,dd] = find(triu(D1)); 
  ne = length(II); 
  S  = sparse(II,1:ne,1,n+m,ne)-sparse(JJ,1:ne,1,n+m,ne);   
  if (m > 0)
     D2 = DD(1:n,n+(1:m)); 
     [I2,J2,d2] = find(D2); 
     if (size(d2,1) < size(d2,2)); d2 = d2'; end
     ne2 = length(I2);
     S2 = sparse(I2,1:ne2,1,n+m,ne2)-sparse(n+J2,1:ne2,1,n+m,ne2);
     S  = [S,S2]; 
     dd = [dd; d2]; 
  end
  Sg = S'; Sg = [Sg(1:length(dd),1:n), sparse(length(dd),m)]; 
  X0 = [X0, anchor]; 
  obj = objfun2(X0,S,dd); 
  gradx = gradfun2(X0,S,dd,Sg);
%%
  Info.objective  = zeros(1,maxit); 
  Info.gradnorm   = zeros(1,maxit); 
  Info.steplength = zeros(1,maxit); 
  Info.objective(1)  = obj; 
  Info.gradnorm(1)   = sqrt(max(sum(gradx.*gradx)));
  
%% 新的
Y =[];
%%N = 10000;

   
delta = 0.2;
Xiter = X0;
objold = obj;
   
for iter = 1:no_first
    objnew = inf;
gradx = gradfun2(Xiter,S,dd,Sg);
alpha = 0.2;
count = 1;
while objnew > objold && count < 20
      alpha = alpha * 0.5;
      count = count + 1;
      Xnew = Xiter - alpha * gradx;
      objnew = objfun2(Xnew,S,dd);
end
Y = [Y,objnew - delta*alpha*norm(gradx(:))^2];
Xiter = Xnew;
objold = objnew;
end

for iter = 1:maxit
objnew = inf;
gradx = gradfun2(Xiter,S,dd,Sg);
alpha = 0.2;
sigma = 0.5;
Xnew = Xiter - alpha * gradx;
objnew = objfun2(Xnew,S,dd);
count = 1;
while objnew > max(Y)
      alpha = alpha * sigma;
      Xnew = Xiter - alpha * gradx;
      objnew = objfun2(Xnew,S,dd);
      count = count + 1;
      if count > 20
          break;
      end
end
Y(1) = [];
Y = [Y,objnew - delta * alpha * norm(Xnew(:))^2];
     Info.objective(iter+1)  = objnew;
     Info.gradnorm(iter+1)   = sqrt(max(sum(gradx.*gradx)));
     Info.steplength(iter+1) = alpha; 
err = abs(objnew - objold)/(1+abs(objold));
if err < tol
objold = objnew;
break;
end 
Xiter = Xnew;
objold = objnew;
end
%%
%   Xnew1 = X0; objold = obj; 
%   objnew1 = objold;
%   for iter = 1:5
%       alpha(iter) = 0.2*0.5^(iter-1);
%       Info.steplength(iter) = alpha(iter); 
%   end
%       gradx1 = gradfun2(Xnew1,S,dd,Sg);
%       Xnew2  = Xnew1 - alpha(1)*gradx1;
%       objnew2 = objfun2(Xnew2,S,dd);
%       gradx2 = gradfun2(Xnew2,S,dd,Sg);
%       Xnew3  = Xnew2 - alpha(2)*gradx2;
%       objnew3 = objfun2(Xnew3,S,dd);
%       gradx3 = gradfun2(Xnew3,S,dd,Sg);
%       Xnew4  = Xnew3 - alpha(3)*gradx3;
%       objnew4 = objfun2(Xnew4,S,dd);
%       gradx4 = gradfun2(Xnew4,S,dd,Sg);
%       Xnew5  = Xnew4 - alpha(4)*gradx4;
%       objnew5 = objfun2(Xnew5,S,dd);
%       
%       
%   for iter = 5:maxit
%       objnew1 = objfun2(Xnew1,S,dd);
%       objnew2 = objfun2(Xnew2,S,dd);
%       objnew3 = objfun2(Xnew3,S,dd);
%       objnew4 = objfun2(Xnew4,S,dd);
%       objnew5 = objfun2(Xnew5,S,dd);
%       objold=[objnew1 objnew2 objnew3 objnew4 objnew5];
%       objold = max(objold);
%      objnew = inf;%不知道怎么？
%      %objold=[objnew1 objnew2 objnew3 objnew4 objnew5];
%      Xiter = Xnew5;
%      gradx = gradfun2(Xiter,S,dd,Sg);
%      for hk=1:10000
%         alpha_k =0.2; thegma = 0.5;delta=0.2;%(此处需按照问题改变)
%          alpha(iter) = alpha_k*thegma*(hk-1);
%          count = 0;
%          while (objnew > objold - delta*alpha(iter).*norm(gradx).^2) && (count < 20)
%             count = count + 1; 
%             Xnew  = Xiter - alpha(iter).*gradx;
%             objnew = objfun2(Xnew,S,dd);
%          end
%      end
%      Xnew1 = Xnew2;
%      Xnew2 = Xnew3;
%      Xnew3 = Xnew4;
%      Xnew4 = Xnew5;
%      Xnew5 = Xnew;
%      
  
  
  if (m > 0); Xiter = Xiter(:,1:n); end
  ttime = etime(clock,ttime); 
  Info.cputime = ttime; 
  Info.objective  = Info.objective(1:iter); 
  Info.gradnorm   = Info.gradnorm(1:iter); 
  Info.steplength = Info.steplength(1:iter); 

%%*************************************************************************
%% Find the function value
%% f(X) = sum_{j<=k} (norm(xj-xk)-djk)^2 + sum_{j,k} (norm(xj-ak)-djk)^2
%%
%% input: X = [sensor position, anchor]
%%*************************************************************************
  function  objval = objfun2(X,S,dd)

  Xij = X*S; 
  normXij = sqrt(sum(Xij.*Xij))';  
  objval = norm(normXij-dd)^2;

%%*************************************************************************
%% Find the gradient of the function 
%% f(X) = sum_{j<=k} (norm(xj-xk)-djk)^2 + sum_{j,k} (norm(xj-ak)-djk)^2
%%
%% input: X = [sensor position, anchor]
%%*************************************************************************
  function  G = gradfun2(X,S,dd,Sg)

  ne = length(dd);
  Xij = X*S;
  normXij = sqrt(sum(Xij.*Xij))'+eps;
  tmp = 1-dd./normXij;
  G = Xij*spdiags(2*tmp,0,ne,ne);
  G = G*Sg;
%%*************************************************************************



