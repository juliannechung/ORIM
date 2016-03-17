%% Paper_Ex2_Deblur.m
%
% Authors:
%   (c) Julianne Chung (e-mail: jmchung@vt.edu)            in March 2016
%       Matthias Chung (e-mail: mcchung@vt.edu)
%
%  This script generates results for Experiment 2 in the paper
%
% [1] Julianne Chung and Matthias Chung. Optimal Regularized Inverse
%     Matrices for Inverse Problems, preprint, 2016.
%
% and requires the RestoreTools package:
%   Nagy, Palmer, and Perrone. Iterative Methods for image deblurring: A
%   Matlab object oriented approach, Numerical Algorithms, 2004.
%

%% Problem setup
rng(0)

load mri;
x_true = double(D(:,:,15));
n = 128;

% Boxcar blur
PSF = zeros(n);
q = 5;
center = [n/2+1, n/2+1];
PSF(center(1)-q:center(1)+q,center(2)-q:center(2)+q) = 1;
PSF = PSF/sum(PSF(:));
A = psfMatrix(PSF, center, 'reflexive');
b = A*x_true(:);

% Add noise to the data
N = randn(size(b(:)));
bn = b(:) + .01*(norm(b(:)) / norm(N(:)))* N;

% Get Tikhonov reconstruction
% PSF is doubly symmetric + reflexive BC -> U and V represent DCT
[U,S,V] = svd(A);
[e, eta] = OptTikTol(U, S, V, reshape(bn,n,n), x_true); % optimal Tikhonov parameter
fprintf('Initial Tikhonov parameter: %.4e\n',eta)
eta2 = eta^2;

phi = reshape(S./(S.^2 + eta2),n,n);
Ztikfun = @(b) reshape(V*(phi.*(U'*reshape(b,n,n))),n*n,1);
P = funMat(Ztikfun, Ztikfun);
x_tik = P*bn;

%% --- Now Compute updates to P to get better reconstructions ---

% (1) ORIM update (M_xi=I, mu_xi~=0)
fprintf('Constructing ORIM update 1 \n')
% Determine mean image - take average of images 8-22 from mri stack
mu = mean( double(D(:,:,[8:14,16:22])), 3);
mu = mu(:);

% Define M = [I, mu_xi]
M = [speye(size(A,2)), mu];

% Get AM as an object
Amu = A*mu;
AMfun = @(x) A*x(1:end-1) + x(end)*Amu ;
AMtfun = @(x) [A'*x; Amu'*x] ;
AMobj = funMat(AMfun, AMtfun);

% Get initial function value
phivec = (S.^2./(S.^2 + eta^2)) - 1;
fprev = (1-eta^2)*norm(phivec,2)^2 + norm(phivec.*(reshape(V'*reshape(mu,n,n),n^2,1)))^2;

% Compute ORIM update - rank 1
param = {'rMax', 1; ...
  'tolAlternating', 1e-6; ...
  'tolIterSolver',1e-6; ...    % default for lsqr 1e-6
  'tolRankUpdate',0*1e-6; ...  % set to zero to ensure all ranks are computed
  'maxIterAlternating', 100;...
  'fprev', fprev;...
  'AM', AMobj}; % set algorithmic parameters
[X_1,Y_1] = orim(A,mu,M,eta,P,param);

% (2) ORIM update (M_xi~=I, mu_xi=0)
% Select covariance matrix to be diag(mu)
M_xi = @(x) sqrt(mu(:)).*x;
M_xit = @(x) sqrt(mu(:)).*x;

% Define M = [M_xi, 0];
Mfun = @(x) M_xi(x(1:end-1));
Mtfun = @(x) [M_xit(x); 0];
Mobj = funMat(Mfun, Mtfun);

% Get AM as an object
AMfun = @(x) A*(M_xi(x(1:end-1))) ;
AMtfun = @(x) [M_xi(A'*x); 0] ;
AMobj = funMat(AMfun, AMtfun);

% Compute ORIM rank-update
param = {'rMax',5; ...
  'tolAlternating', 1e-6; ...
  'tolIterSolver',1e-6; ...    % default for lsqr 1e-6
  'tolRankUpdate',0*1e-6; ...  % set to zero to ensure all ranks are computed
  'maxIterAlternating', 100;...
  'fprev', fprev;...
  'AM', AMobj}; % set algorithmic parameters

[X_2,Y_2] = orim(A,zeros(size(mu)),Mobj,eta,P,param);

% (3) ORIM update (M_xi~=I, mu_xi~=0)

% Define M = [M_xi,mu_xi]
Mfun = @(x) M_xi(x(1:end-1)) + mu*x(end);
Mtfun = @(x) [M_xit(x); mu'*x];
Mobj = funMat(Mfun, Mtfun);

% Get AM object
AMfun = @(x) A*(M_xi(x(1:end-1))) + x(end)*Amu ;
AMtfun = @(x) [M_xit(A'*x); Amu'*x] ;
AMobj = funMat(AMfun, AMtfun);

% Compute ORIM rank-update
param = {'rMax',5; ...
  'tolAlternating', 1e-6; ...
  'tolIterSolver',1e-6; ...    % default for lsqr 1e-6
  'tolRankUpdate',0*1e-6; ...  % set to zero to ensure all ranks are computed
  'maxIterAlternating', 100;...
  'fprev', fprev;...
  'AM', AMobj}; % set algorithmic parameters

[X_3,Y_3] = orim(A,mu,Mobj,eta,P,param);

%% Compute Reconstructions
x_orim1 = x_tik + X_1*(Y_1'*bn(:));
x_orim2 = x_tik + X_2*(Y_2'*bn(:));
x_orim3 = x_tik + X_3*(Y_3'*bn(:));

%% Display images
ntrue = norm(x_true(:));

figure, 
subplot(2,2,1), imshow(x_true,[]), title('True')
subplot(2,2,2), imshow(reshape(bn,size(x_true)),[]), title('Blurred')
subplot(2,2,3), imshow(PSF,[]), title('PSF')
subplot(2,2,4), imshow(reshape(mu,size(x_true)),[]), title('mean image')

figure,
subplot(2,2,1), imshow(reshape(x_tik,n,n),[]), title(sprintf('Tikhonov, rel=%.4f',norm(x_tik-x_true(:))/ntrue))
subplot(2,2,2), imshow(reshape(x_orim1,n,n),[]), title(sprintf('ORIM update 1, rel=%.4f',norm(x_orim1-x_true(:))/ntrue))
subplot(2,2,3), imshow(reshape(x_orim2,n,n),[]), title(sprintf('ORIM update 2, rel=%.4f',norm(x_orim2-x_true(:))/ntrue))
subplot(2,2,4), imshow(reshape(x_orim3,n,n),[]), title(sprintf('ORIM update 3, rel=%.4f',norm(x_orim3-x_true(:))/ntrue))

% Error Images
err1 = abs(x_tik-x_true(:));
err2 = abs(x_orim3-x_true(:));
maxval = max([max(err1),max(err2)]);
figure, 
subplot(1,2,1), imshow(reshape(err1,n,n),[0,maxval]), colormap(flipud(colormap)), title('Tikhonov')
subplot(1,2,2), imshow(reshape(err2,n,n),[0,maxval]), colormap(flipud(colormap)), title('ORIM update 3')

