function [Z,f,Uh,Sh,X,Y] = orimTheorem(A,M,P,eta2,maxJ,objfun)
%
% function [Z,f,Uh,Sh,X,Y] = orimTheorem(A,M,P,eta2,maxJ,objfun)
%
% Authors:
%   (c) Julianne Chung (e-mail: jmchung@vt.edu)            in February 2016
%       Matthias Chung (e-mail: mcchung@vt.edu)
%
% MATLAB Version: 8.6.0.267246 (R2015b)
%
% Description:
%   This function computes the generalized optimal inverse matrix of rank r
%
%     Z = argmin_{rank(Z) <= r} f(Z) = || Z [AM   eta I] - [M-PAM    -eta P] ||_F
%
%   using a closed form solution, see [1]. Note, condition
%
%       rank((I - P*A)M*M'*A + eta^2*P) >= r
%
%   must hold. Notice also that the code is not optimized for large scale
%   problems.
%
% Input arguments:
%   A      -  matrix [m x n] of rank k, m >= n
%   M      -  matrix [n x p] with p >= n, rank(M) = n
%   P      -  matrix [n x m]
%   eta2   -  eta^2
%   r      -  integer r <= k
%   objfun - provide the objective function of interest otherwise f
%
% Output arguments:
%   Z      - global minimizer of f
%   f      - global minimum of f for each rank
%   Uh     - global minimizer of f
%   Sh     - global minimizer of f
%   X      - provide decomposition X of Z = XY'
%   Y      - provide decomposition Y of Z = XY'
%
% References:
%    [1] Julianne Chung and Matthias Chung. Optimal Regularized Inverse
%        Matrices for Inverse Problems, preprint, 2016.
%

[m,n] = size(A);                                            % get size of A
p = size(M,2);                                              % get # of columns of M

if nargin < 6 % define objective function if required
  objfun = @(Zhat)norm(((Zhat+P)*A- eye(n))*M,'fro')^2 + eta2*norm(Zhat+P,'fro')^2;
end

[U,~,Ginv,Sigma,S] = gsvd(A,M');                            % get GSVD of {A,M'}
Ginv = Ginv';                                               % match notation of paper
L = Sigma*(Ginv*(Ginv'*S'));                                % define L
[Ul,Sigmal,~] = svd(L);                                     % get SVD of L
Dam = [sqrt(diag(Sigmal).^2+eta2); sqrt(eta2)*ones(m-p,1)]; % define Dam
Dam2 = Dam.^2;                                              % define Dam^2
F = (eye(n) - P*A)*M*(M'*A')*U*Ul - eta2*P*U*Ul;            % define F
H = F*diag(1./Dam2)*F';                                     % define H
[Uh, Sh,~] = svd(H);                                        % get SVD/eigenvalue decomposition of H
T = F*diag(1./Dam2)*Ul'*U';                                 % define T

if nargout > 1
  f = zeros(maxJ,1);
  for r = 1:maxJ
    Z    = Uh(:,1:r)*Uh(:,1:r)'*T;                          % define ORIM
    f(r) = objfun(Z);                                       % calculate function value
  end
  X = Uh(:,1:maxJ);                                         % compute decomposition X
  Y = T'*Uh(:,1:maxJ);                                      % compute decomposition Y
else
  Z = Uh(:,1:maxJ)*Uh(:,1:maxJ)'*T;                         % define ORIM
  X = Uh(:,1:maxJ);                                         % compute decomposition X
  Y = T'*Uh(:,1:maxJ);                                      % compute decomposition Y
end

end