%% EXAMPLE 1 (Study 1) of [1] Paper_Ex1_1.m
%
% Authors:
%   (c) Julianne Chung (e-mail: jmchung@vt.edu)            in February 2016
%       Matthias Chung (e-mail: mcchung@vt.edu)
%
%  This script generates results for Example 1 (study 1) in the paper
%
% [1] Julianne Chung and Matthias Chung. Optimal Regularized Inverse
%     Matrices for Inverse Problems, preprint, 2016.
%
% and requires reg tools
%
%     www.mathworks.com/matlabcentral/fileexchange/52-regtools)
%

%% set up problem and generate matrix A, M, P and parameter eta
rng(0)                                   % set random generator seed
n = 1000;                                % dimension of problem m = n and p = n+1
A = heat(n);                             % set matrix A (requires regtools)
M = randn(n,n+1);                        % set matrix M with condition rank(M) = n
P = randn(n,n);                          % set matrix P
eta = 0.02; eta2 = eta^2;                % set noise level parameter

%% compute ORIM using update approach

% set parameters for ORIM
param = {'rMax',                 n; ...
         'tolAlternating',    1e-6; ...  
         'tolIterSolver',     1e-8; ...  
         'tolRankUpdate',        0; ...  % set to zero to ensure all ranks are computed
         'maxIterAlternating', 200};
       
[X,Y,~,f_update] = orim(A,M(:,end),M(:,1:end-1),eta,P,param);
Z_update = X*Y';

%% compute ORIM matrix via Theorem 3.3, compare ORIM (update), TSVD, TTik, and ORIM (Theorem) for various ranks r

% define objective function 
objfun = @(Zhat)norm(((Zhat+P)*A - eye(n))*M,'fro')^2 + eta2*norm(Zhat+P,'fro')^2;

[U,S,V] = svd(A); s = diag(S); % compute SVD of A for TSVD solution

f_TSVD = zeros(n,1); f_TTIK = f_TSVD; % initialize 

h = waitbar(0,'Computing ranks ...');

% get results for each rank
for r = 1:n
  
  waitbar(r/n)
  
  % TSVD for A
  Z_TSVD = V(:,1:r)*diag(1./s(1:r))*U(:,1:r)';
  f_TSVD(r) = objfun(Z_TSVD);
  
  % TTik for A
  psi = s(1:r)./(s(1:r).^2 + eta2);
  Z_TTIK = V(:,1:r)*diag(psi)*U(:,1:r)';
  f_TTIK(r) = objfun(Z_TTIK);
  
end

close(h)

%% compute ORIM from Theorem and ORIM_0

fprintf('Computing ORIMs via Theorem 3.3 ... ')
[Zhat,f,Uh,Sh] = orimTheorem(A,M,P,eta2,n,objfun);                                      % compute ORIM
[Z,f_orim]     = orimTheorem(A,[M(1:n,1:n), zeros(n,1)],zeros(size(A')),eta2,n,objfun); % compute ORIM_0
fprintf('done.\n')

%% plot results and compute relative error between rank update approach and direct computation

figure(1)
plot(f_TSVD,':',  'LineWidth',2,'color',[0.9290 0.6940 0.1250]), hold on,
plot(f_TTIK,'--', 'LineWidth',2,'color',[0.4940 0.1840 0.5560])
plot(f_orim,'-.', 'LineWidth',2,'color',[0.4660 0.6740 0.1880])
plot(f_update,'-','LineWidth',2,'color',[0      0.4470 0.7410])

fontSize = 18;
hLegend  = legend('TSVD', 'TTik', 'ORIM_0', 'ORIM'); set(hLegend,'FontSize',fontSize);
xlabel('rank $r$','Interpreter','Latex','FontSize',fontSize)
ylabel('objective function $f(\mathbf{\widehat Z}_r)$','Interpreter','Latex','FontSize',fontSize)
set(gca,'FontSize',fontSize);
set(gca,'FontName','Times New Roman')
axis([1 50 9.2e5 1.28e6])  
box off

fprintf('The maximal absolute relative error is %1.4e \n',max(abs(f-f_update)./f))
