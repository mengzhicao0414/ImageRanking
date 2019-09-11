function [Y, y, infos] = ENewton(D, eigsolver, prnt, y, error_tol)
%% function [X, y, infos] = ENewton(D, y)
% Input: D  -- Squared predistance matrix (diag(D) =0)
%        y  -- starting point, a vector of length(D)
%              default: y = zeors(n,1);
%  eigsolver - =1 eig.m
%            - =0 mexeig.m
%     prnt  -- 1 print information
%              0 no output information
% error_tol -- termination tolerance default: 1.0e-6
% Output:
%      Y -- Nearest Euclidean distance matrix from D
%      y -- the optimal Lagrange multiplier corresponding to the diagonal
%           constraints
%    infos -- information about Y
%             infos.P = P;
%         infos.lambda = lambda;
%         infos.rank = Imbedding_Dimension;
%         infos.Iter = k;
%         infos.feval = f_eval; number of function evaluations;
%         infos.t = time_used; total cpu used
%         infos.res = norm_b; norm of the gradient at the final iterate
%                             This is the residual one finally got.
%         infos.f = val_obj; final objective function value 
%%
%
%  This code is designed to solve %%%%%%%%%%%%%
%   min 0.5*<X-D, X-D>
%   s.t. X_ii =0, i=1,2,...,n
%        X is positive semidefinite on the subspace
%          \{x\in \Re^n: eTx =0 \}
%%
%  Based on the algorithm  in %%%%%
%  ``A Semismooth Newton Method for 
%  the Nearest Euclidean Distance Matrix Problem'' 
%                By 
%             Houduo Qi                        
%   
%  First version date:  August 16, 2011
%  Last modified date:  August 26 2011  
%%          
% Send your comments and suggestions to    %%%%%%
%        hdqi@soton.ac.uk      %%%%%%
%
% Acknowledgment: The code makes use of CorNewton.m developed by
% Houduo Qi and Defeng Sun (NUS) for computing 
% the nearest correlation matrix 
%
%%%%% Warning: Accuracy may not be guaranteed!!!!! %%%%%%%%
%%
%
t0 = tic;
n = length(D);
% 
if nargin < 3
    prnt = 1;
end
if nargin < 4
   y = zeros(n,1);
end
if nargin < 5
   error_tol = 1.0e-6;
end
%
if prnt
   fprintf('\n ******************************************************** \n')
   fprintf( '          The Semismooth Newton-CG Method                ')
   fprintf('\n ******************************************************** \n')
   fprintf('\n The information of this problem is as follows: \n')
   fprintf(' Dim. of    sdp      constr  = %d \n',n)
end

D = -(D+D')/2; % make D symmetric
               % use -D instead because it is (-D) which is psd on e^\perp
%
% calculate JDJ
%
De  = sum(D, 2); % row sums
eDe = sum(De); % total sum of D
JDJ = -De*(ones(1,n)/n);
JDJ = JDJ + JDJ';
JDJ = JDJ + D + eDe/n^2;
%JDJ = (JDJ + JDJ')/2; % make JDJ symmetric
%
Fy = zeros(n,1);
k=0;
f_eval = 0;
Iter_Whole = 200;
Iter_inner = 20; % Maximum number of Line Search in Newton method
maxit = 200; %Maximum number of iterations in PCG
iterk = 0;
Inner = 0;
tol = 1.0e-2; %relative accuracy for CGs
%
sigma_1=1.0e-4; %tolerance in the line search of the Newton method
%
x0=y;
%
prec_time = 0;
pcg_time = 0;
eig_time =0;

c = ones(n,1);
%M = diag(c); % Preconditioner to be updated
%
d =zeros(n,1);
val_G = sum(sum(D.*D))/2;
%
% calculate Y = - J(D+diag(y))J
%
Y = JyJ(y);
Y = - (JDJ + Y);
Y = (Y+Y')/2;
%%%%% eigendecomposition
 eig_time0 = tic;
%%% there are two ways to do this
%%% one is to use matlab: eig
%%% the other is to use mexeig (64-bit) written by Defeng
%%% use matlab eig.m
if eigsolver %=1 use eig.m
  [P, Lambda] =  eig(Y);   %% X= P*diag(D)*P'
  P = real(P);
  lambda = real(diag(Lambda));
else % use mexeig
 [P,Lambda] = mexeig(Y);   %% X= P*diag(D)*P'
  P = real(P);
 lambda = real(Lambda);
end
 eig_time = eig_time + toc(eig_time0); 
%%% end of eigendecompsition

 if lambda(1) < lambda(n) %the eigenvalues are arranged in the decreasing order
     lambda = lambda(n:-1:1);
    P = P(:,n:-1:1);
 end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 [f0,Fy] = gradient(y,D,lambda,P,n);
 f = f0;
 f_eval = f_eval + 1; % number of function evaluations increased by 1
 b = - Fy;
 norm_b = norm(b);

 Initial_f = val_G - f0;
if prnt
 fprintf('Initial Dual Objective Function value==== %d \n', Initial_f)
 fprintf('Newton: Norm of Gradient %d \n',norm_b)
end

 Omega12 = omega_mat(lambda,n);
 x0 = y;

 tt = toc(t0);
[hh,mm,ss] = time(tt);

if prnt
fprintf('\n   Iter.   Num. of CGs     Step length      Norm of gradient     time_used ')
fprintf('\n    %d         %2.0d            %3.2e                      %3.2e         %d:%d:%d ',0,str2num('-'),str2num('-'),norm_b,hh,mm,ss)
end

 while (norm_b>error_tol & k< Iter_Whole)

  prec_time0 = tic;
   c = precond_matrix(Omega12,P,n); % comment this line for  no preconditioning
  prec_time = prec_time + toc(prec_time0);
  
 pcg_time0 = tic;
 [d,flag,relres,iterk]  =pre_cg(b,tol,maxit,c,Omega12,P,n);
 pcg_time = pcg_time + toc(pcg_time0);
 %d =b0-Fy; gradient direction
 %fprintf('Newton: Number of CG Iterations %d \n', iterk)
  
  if (flag~=0); % if CG is unsuccessful, use the negative gradient direction
     % d =b0-Fy;
     disp('..... Not a full Newton step......')
  end
 slope = (Fy)'*d; %%% nabla f d
 

    y = x0 + d; %temporary x0+d  
      Y = JyJ(y);
      Y = - (JDJ + Y);
      Y = (Y+Y')/2;
%%%%% eigendecomposition
 eig_time0 = tic;
%%% there are two ways to do this
%%% one is to use matlab: eig
%%% the other is to use mexeig (64-bit) written by Defeng
if eigsolver %=1 use eig.m
  [P, Lambda] =  eig(Y);   %% X= P*diag(D)*P'
  P = real(P);
  lambda = real(diag(Lambda));
else % use mexeig
 [P,Lambda] = mexeig(Y);   %% X= P*diag(D)*P'
  P = real(P);
 lambda = real(Lambda);
end

 eig_time = eig_time + toc(eig_time0); 
%%% end of eigendecompsition
 if lambda(1) < lambda(n) %the eigenvalues are arranged in the decreasing order
     lambda = lambda(n:-1:1);
    P = P(:,n:-1:1);
 end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     [f,Fy] = gradient(y,D,lambda,P,n); % increase of f_eval will be added
                                        % after the linear search  
     k_inner = 0;
     while(k_inner <=Iter_inner & f> f0 + sigma_1*0.5^k_inner*slope + 1.0e-6)
                           % line search procedure
         k_inner = k_inner+1;
         y = x0 + 0.5^k_inner*d; % backtracking   
         
        Y = JyJ(y);
        Y = - (JDJ + Y);
        Y = (Y+Y')/2;
         
        %%%%% eigdecomposition
 %%%%% eigendecomposition
 eig_time0 = tic;
%%% there are two ways to do this
%%% one is to use matlab: eig
%%% the other is to use mexeig (64-bit) written by Defeng
if eigsolver %=1 use eig.m
  [P, Lambda] =  eig(Y);   %% X= P*diag(D)*P'
  P = real(P);
  lambda = real(diag(Lambda));
else % use mexeig
 [P,Lambda] = mexeig(Y);   %% X= P*diag(D)*P'
  P = real(P);
 lambda = real(Lambda);
end
%
 eig_time = eig_time + toc(eig_time0); 
%%% end of eigendecompsition
 if lambda(1) < lambda(n) %the eigenvalues are arranged in the decreasing order
     lambda = lambda(n:-1:1);
    P = P(:,n:-1:1);
 end 
%%% this part is for mexeig.m
%  lambda = (real(Lambda));
%  P = real(P);
%   if lambda(1) < lambda(n) %the eigenvalues are arranged in the decreasing order
%      lambda = lambda(n:-1:1);
%     P = P(:,n:-1:1);
%  end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         [f,Fy] = gradient(y,D,lambda,P,n);
      end % end of the line search procedure
      if k_inner >=1
        fprintf('\n number of linear serach: %d', k_inner)
      end
      f_eval = f_eval + k_inner + 1; % number of function evaluations
                                     % is increased over the line search
                                     % and the one before it
      x0 = y;
      f0 = f;
      
     k=k+1;
     b = -Fy;
     norm_b = norm(b);
     tt = toc(t0);
    [hh,mm,ss] = time(tt); 
  %   fprintf('Newton: Norm of Gradient %d \n',norm_b)
if prnt      
    fprintf('\n   %2.0d         %2.0d             %3.2e          %3.2e         %d:%d:%d', ...
        k,iterk,0.5^k_inner,norm_b,hh,mm,ss)
end
     Res_b(k) = norm_b;
    
     Omega12 = omega_mat(lambda,n);

 end %end loop for while i=1;
Ip = find(lambda>0); % could set to 1.0e-7
In = find(lambda<-1.0e-8); % could set to -1.0e-7
Imbedding_Dimension = length(In);
  % The eigen-decomposition is on -J(D+\A^*y)J
  % The imbedding dimension is the number of negative eigenvalues of
  % -J(D+\A^*y)J.
  % This result is based on Haydan-Wells projection formula
  % see Eq. (37) in my paper.
r = length(Ip);
 
if (r==0)
    Y = D + diag(y);
elseif (r==n)
    Y = D + diag(y) + Y;
elseif (r<=n/2)
    lambda1 = lambda(Ip);
    lambda1 = lambda1.^0.5;
    P1 = P(:, 1:r);
    P1 = P1*sparse(diag(lambda1));
    Y = P1*P1';
    Y = D+diag(y) + Y;% Optimal solution X* 
else 
    
    lambda2 = -lambda(r+1:n);
    lambda2 = lambda2.^0.5;
    P2 = P(:, r+1:n);
    P2 = P2*sparse(diag(lambda2));
    Y = Y + P2*P2'; 
    Y = D+diag(y) + Y;% Optimal solution X* 
end
 Y = (Y+Y')/2;
 Final_f = val_G-f;
 %
 % set diagonals of Y to zero
 %
 i=1;
 while i<=n
     Y(i,i) = 0;
     i=i+1;
 end
 %
 val_obj = sum(sum((Y-D).*(Y-D)))/2;
 time_used = toc(t0); 
% fprintf('\n')
% set output information
infos.P = P;
infos.lambda = lambda;
infos.rank = Imbedding_Dimension;
infos.Iter = k;
infos.feval = f_eval;
infos.t = time_used;
infos.res = norm_b;
infos.f = val_obj;
infos.dualZ = D+diag(y) - Y; % optimal dual solution in \K^*
infos.dualy = y;
%
 % Put the sign (-1) back
 %
 Y = -Y;
%
if prnt
fprintf('\n\n')
fprintf('Norm of Gradient %d \n',norm_b)
fprintf('Number of Iterations == %d \n', k)
fprintf('Number of Function Evaluations == %d \n', f_eval)
fprintf('Final Dual Objective Function value ========== %d \n',Final_f)
fprintf('Final Original Objective Function value ====== %d \n', val_obj)
fprintf('Imbedding dimension ================= %d \n',Imbedding_Dimension)

fprintf('Computing time for computing preconditioners == %d \n', prec_time)
fprintf('Computing time for linear systems solving (cgs time) ====%d \n', pcg_time)
fprintf('Computing time for  eigenvalue decompostions (calling eig time)==%d \n', eig_time)
fprintf('Total computing time (in s) ==== =====================%d \n',time_used)
end
% 


%%% end of the main program

%%% To change the format of time 
function [h,m,s] = time(t)
t = round(t); 
h = floor(t/3600);
m = floor(rem(t,3600)/60);
s = rem(rem(t,60),60);
return
%%% End of time.m

%%%%%% To generate J*diag(y)J
%
function Y = JyJ(y)

n = length(y);
Y = - y*(ones(1,n)/n);
Y = Y+Y';
Y = Y + diag(y) + sum(y)/n^2;

return

%%%%%%
%%%%%% To generate F(y) %%%%%%%
%%%%%%%

function [f,Fy]= gradient(y,D,lambda,P,n)
 
%[n,n]=size(P);
 f = 0.0;
Fy = zeros(n,1);
 
%lambdap=max(0,lambda);
%H =diag(lambdap); %% H =P^T* H^0.5*H^0.5 *P

P=P';
 i=1;
 while (i<=n)
     P(i,:)=max(lambda(i),0)^0.5*P(i,:);
     i=i+1;
 end
 i=1;
 while (i<=n)
       Fy(i) = P(:,i)'*P(:,i) + y(i) + D(i,i);
       i=i+1;     
 end
  
 Dy = D + diag(y);
 
 % Compute the objective function (because it is easy to calculate)
%
In = find(lambda<0); % negative eigenvalues
lambdan = lambda(In);
f = f + lambdan'*lambdan;
%
% the rest part depends on Q
%
Dye = sum(Dy, 2); % Dy*e
%
% to save using a new variable yhat (see below)
%
c = (sum(Dye) + sqrt(n)*Dye(n))/(n+sqrt(n));
Dye = (-Dye + c)/sqrt(n); %  = -Dye/sqrt(n) + c/sqrt(n);
                          % denoted by Dye (Dye not to be used any more)
Dye(end) = Dye(end) + c;
 
 f = f + sum(Dye.^2) + sum(Dye(1:end-1).^2);
 f = 0.5*f;

%%%%%%%%%%%%%%%%%%%%%%%%% 
% yhat = zeros(n,1);
% 
% c = (sum(Dye) + sqrt(n)*Dye(n))/(n+sqrt(n));
% yhat = (-Dye + c)/sqrt(n); %  = -Dye/sqrt(n) + c/sqrt(n);
% yhat(n) = yhat(n) + c;
% 
% f = f + yhat'*yhat + yhat(1:n-1)'*yhat(1:n-1);
% f = 0.5*f;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Proj = zeros(n,n); % denote the projection matrix of -J(D+diag(y))J onto
%                    % positive semidefinite cone
% % calculate the upper off diagonal elements
% %
% for i=1:n-1
%     for j=i+1:n
%         Proj(i,j) = P(:,i)'*P(:, j);
%     end
% end
% Proj = Proj + Proj';
% %
% % calculate the diagonal element of Proj
% % and the gradient Fy
% %
%  i=1;
%  while (i<=n)
%        Proj(i,i) = P(:,i)'*P(:,i);
%        Fy(i) = Proj(i,i) + y(i);
%  i=i+1;     
%  end
%  %
%  % calculate the objective function
%  %
%  Proj = Proj + D + diag(y);
%  Proj = Proj.*Proj;
%  f = 0.5*sum(sum(Proj));
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% end of gradient.m %%%%%%

%%%%%%%%%%%%%%        %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% To generate the first -order difference of lambda
%%%%%%%

%%%%%%%%%%%%%%        %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% To generate the essential part of the first -order difference of d
%%%%%%%
function Omega12 = omega_mat(lambda,n)
%We compute omega only for 1<=|idx|<=n-1
idx.idp = find(lambda>0);
idx.idm = setdiff([1:n],idx.idp);
n =length(lambda);
r = length(idx.idp);
 
if ~isempty(idx.idp)
    if (r == n)
        Omega12 = ones(n,n);
    else
        s = n-r;
        dp = lambda(1:r);
        dn = lambda(r+1:n);
        Omega12 = (dp*ones(1,s))./(abs(dp)*ones(1,s) + ones(r,1)*abs(dn'));
        %  Omega12 = max(1e-15,Omega12);

    end
else
    Omega12 =[];
end

    %%***** perturbation *****
    return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% end of omega_mat.m %%%%%%%%%%

%%%%%% PCG method %%%%%%%
%%%%%%% This is exactly the algorithm by  Hestenes and Stiefel (1952)
%%%%%An iterative method to solve A(x) =b  
%%%%%The symmetric positive definite matrix M is a
%%%%%%%%% preconditioner for A. 
%%%%%%  See Pages 527 and 534 of Golub and va Loan (1996)

function [p,flag,relres,iterk] = pre_cg(b,tol,maxit,c,Omega12,P,n);
% Initializations
r = b;  %We take the initial guess x0=0 to save time in calculating A(x0) 
n2b =norm(b);    % norm of b
tolb = tol * n2b;  % relative tolerance 
p = zeros(n,1);
flag=1;
iterk =0;
relres=1000; %%% To give a big value on relres
% Precondition 
z =r./c;  %%%%% z = M\r; here M =diag(c); if M is not the identity matrix 
rz1 = r'*z; 
rz2 = 1; 
d = z;
% CG iteration
for k = 1:maxit
   if k > 1
       beta = rz1/rz2;
       d = z + beta*d;
   end
   %w= Jacobian_matrix(d,Omega,P,n); %w = A(d); 
   w = Jacobian_matrix(d,Omega12,P,n); % W =A(d)
   denom = d'*w;
   iterk =k;
   relres = norm(r)/n2b;              %relative residue = norm(r) / norm(b)
   if denom <= 0 
       sssss=0
       p = d/norm(d); % d is not a descent direction
       break % exit
   else
       alpha = rz1/denom;
       p = p + alpha*d;
       r = r - alpha*w;
   end
   z = r./c; %  z = M\r; here M =diag(c); if M is not the identity matrix ;
   if norm(r) <= tolb % Exit if Hp=b solved within the relative tolerance
       iterk =k;
       relres = norm(r)/n2b;          %relative residue =norm(r) / norm(b)
       flag =0;
       break
   end
   rz2 = rz1;
   rz1 = r'*z;
end

return

%%%%%%%% %%%%%%%%%%%%%%%
%%% end of pre_cg.m%%%%%%%%%%%


%%% To generate the Jacobian product with x: F'(y)(x)
function Ax = Jacobian_matrix(x,Omega12,P,n)

[r,s]  = size(Omega12); 
Ax = zeros(n,1);

if (r==0)
    Ax = (1 + 1.0e-10)*x;
elseif (r==n)
    Ax = (2/n+1.0e-10)*x - sum(x)/n^2;
else
    P1 = P(:,1:r);
    P2 = P(:,r+1:n);
    
    %PT = P'; % PT can be saved to save memory
    pbar = sum(P)'/n; %sum(P', 2)/n;
    sumx = sum(x);
    startx = P'*x;
    
        if (r<n/2)
            %H = (Omega.*(P'*sparse(Z)*P))*P';
            H1 = P1'*sparse(diag(x));
            Omega12_old = Omega12;
            Omega12 = Omega12.*(H1*P2);
            H = [(H1*P1)*P1' + Omega12*P2'; Omega12'*P1'];
           
            i=1;
            while (i<=n)
                Ax(i) = P(i,:)*H(:,i); % part from the digonal part
                pi = P(i,:)'.*pbar;
                pj = P(i,:)'.*(startx - 0.5*sumx*pbar);
                v  = sum(pi(1:r))*sum(pj(1:r)) ...
                     + pi(1:r)'*Omega12_old*pj(r+1:n) ...
                     + pi(r+1:n)'*Omega12_old'*pj(1:r);
                 
                Ax(i) = Ax(i) - 2*v;
                Ax(i) = x(i) - Ax(i);
                Ax(i) = Ax(i) + 1.0e-10*x(i);   
                i=i+1;
            end
         else % if r>=n/2, use a complementary formula.
            %H = ((E-Omega).*(P'*Z*P))*P';               
            H2 = P2'*sparse(diag(x));
            Omega12 = ones(r,s)- Omega12;
            Omega12_old = Omega12;
            Omega12 = Omega12.*((H2*P1)');
            H = [Omega12*P2'; Omega12'*P1' + (H2*P2)*P2'];
           
            i=1;
            while (i<=n)
               Ax(i) = x(i) - P(i,:)*H(:,i); %from diagonal part
               pi = P(i,:)'.*pbar;
               pj = P(i,:)'.*(startx - 0.5*sumx*pbar);
               v  = sum(pi(r+1:n))*sum(pj(r+1:n)) ...
                     + pi(1:r)'*Omega12_old*pj(r+1:n) ...
                     + pi(r+1:n)'*Omega12_old'*pj(1:r);
               v = sum(pi)*sum(pj) - v;
                 
               Ax(i) = Ax(i) - 2*v;
               Ax(i) = x(i) - Ax(i);
               x(i) = Ax(i) + 1.0e-10*x(i);   
               i=i+1;
            end
        end
end
return
%%% End of Jacobian_matrix.m  
   

%%%%%% To generate the diagonal preconditioner%%%%%%%
%%%%%%%

function c = precond_matrix(Omega12,P,n)

[r,s] =size(Omega12);
c = ones(n,1);

if (r>0)

    if (r< n/2)
        H = P';
        h = sum(H,2)/n; %average row sum
        H = H.*(H-h(:, ones(n,1))); % H.*(H-[h,h,...,h])

        H12 = H(1:r,:)'*Omega12;
        d =ones(r,1);
        for i=1:n
            c(i) = sum(H(1:r,i))*(d'*H(1:r,i));
            c(i) = c(i) +2.0*(H12(i,:)*H(r+1:n,i));
            c(i) = 1 - c(i);
            if c(i) < 1.0e-8
                c(i) =1.0e-8;
            end
        end
    else
         if r<n% if r>=n/2, use a complementary formula
        H = P';
        h = sum(H,2)/n; %average row sum
        H = H.*(H-h(:, ones(n,1))); % H.*(H-[h,h,...,h])
        
        Omega12 = ones(r,s)-Omega12;
        H12 = Omega12*H(r+1:n,:);
        d =ones(s,1);
        dd = ones(n,1);
        
        for i=1:n
            c(i) = sum(H(r+1:n,i))*(d'*H(r+1:n,i));
            c(i) = c(i) + 2.0*(H(1:r,i)'*H12(:,i));
            alpha = sum(H(:,i));
            c(i) = alpha*(H(:,i)'*dd)-c(i);
            c(i) = 1- c(i);
            if c(i) < 1.0e-8
                c(i) =1.0e-8;
            end
        end

         end
    end
end

return

 
%%%%%%%%%%%%%%%
%end of precond_matrix.m%%%

%



