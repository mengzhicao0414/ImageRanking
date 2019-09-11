% made by yupanpan
% this code is designed to calculate spectral decomposition of A and make A positive semidefinite
% calculate spectral decomposition of A:
% A=UVU^T
% to make A positive semidefinite: A = U max(V,0) U^T 

% input:
% A            the distance metric
% d_train      the dimension of training data

% output:
% PSDA         positive semidefinite matrix of A
function[PSDA]=PSD(A,d_train)
A=A+A';  
A=A*0.5; 
[PSDU,PSDV]=eig(A);
    for i=1:d_train
        if PSDV(i,i)<0
         PSDV(i,i)=0;   
        end
    end
PSDA=PSDU*PSDV*PSDU';
end
