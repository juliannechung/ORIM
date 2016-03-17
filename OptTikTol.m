function [e, alpha] = OptTikTol(U, S, V, b, x_true)
%
%   [e, alpha] = OptTolTol(U, S, V, b, x)
%
%   This function computes the optimal regularization parameter
%       for Tikhonov regularization by minimizing:
%           || filtered solution - true solution ||_2
%   
%   Assume the problem has the form b = Ax + noise, with [U, S, V] = svd(A).
%
%   Input:
%       [U, S, V] - SVD of A, where S is a column vector containing
%                       singular or spectral values of A
%               b - noisy data (come in as images)
%               x - true data (come in as images)
%
%   Output:
%               e - relative error
%           alpha - optimal regularization parameter

bhat = U'*b;
bhat = bhat(:);
alpha = fminbnd('TikRelErrors',0,1,[],bhat,V'*x_true,S);
e = TikRelErrors(alpha, bhat, V'*x_true, S);



function E = TikRelErrors(alpha, bhat, xhat, S)
%
%   E = TikRelErrors(alpha, bhat, xhat, S)
%
%   This function computes the relative error for Tikhonov Regularization.
%      
%   Assume the problem has the form b = Ax + noise, with [U, S, V] = svd(A).
%
%   Input:
%       alpha - regularization parameter
%        bhat - U'*b, where b is the noisy data
%        xhat - V'*x_true
%           S - column vector containing singular or spectral values of A
%   Output:
%       E - relative error at the given alpha
%

bhat = bhat(:);
xhat = xhat(:);
y = ( (abs(S).^2) .* bhat ./ S ) ./ ( abs(S).^2 + alpha^2 ) - xhat;
E = sum(abs(y).^2);
