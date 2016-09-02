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

%% set up problem generating matrix A, M, P and parameter etc
rng(100)
m = 1000; n = m; p = n+1;                % dimension of problem m > n and p >= n
[A,b,x] = heat(n);                       % get heat matrix A (requires regtools)
k = 50;                                  % choose max rank to k = 50
M = randn(n,p); M(:,end) = 5*M(:,end);   % set matrix M with condition rank(M) = n
P = randn(n,m);                          % set matrix P
eta = 0.1; eta2 = eta^2;                 % set noise level parameter

%% set parameters for ORIM method
param = {'rMax',                 k;  ...  % compute ranks up to k
         'tolAlternating',    1e-6;  ...  % use alternating direction tolerance to 1e-6
         'tolIterSolver',     1e-8;  ...  % use LSQR solver tolerance to 1e-8
         'tolRankUpdate',        0;  ...  % set to zero to ensure all ranks are computed
         'maxIterAlternating', 200}; ...  % set algorithmic parameters
         
%% compute ORIM using update approach        
[X,Y,~,f_update] = orim(A,M(:,end),M(:,1:end-1),eta,P,param);
Z_update = X*Y';

%% compute TSVD, and TTik for various ranks r


objfun = @(Zhat)norm(((Zhat+P)*A- eye(n))*M,'fro')^2 + eta2*norm(Zhat+P,'fro')^2; % define objective function

[U,S,V] = svd(A); s = diag(S); % compute SVD for TSVD solution

f_TSVD = zeros(k,1); f_TTIK = f_TSVD; % initialize function values

h = waitbar(0,'Computing ranks ...');

% iterate over ranks
for r = 1:k
  
  waitbar(r/k)
  
  % TSVD for A
  Z_TSVD = V(:,1:r)*diag(1./s(1:r))*U(:,1:r)';
  f_TSVD(r) = objfun(Z_TSVD);
  
  % TTik for A
  psi = s(1:r)./(s(1:r).^2 + eta2);
  Z_TTIK = V(:,1:r)*diag(psi)*U(:,1:r)';
  f_TTIK(r) = objfun(Z_TTIK);
  
end

close(h)

%% compute ORIM and ORIM_0 via Theorem 3.3 and 
fprintf('Computing ORIM using Theorem 3.3 ...                  ')
[Zhat,f,Uh,Sh] = orimTheorem(A,M,P,eta2,k,objfun); fprintf('done.\n')

fprintf('Computing ORIM using Theorem 3.3 (P = 0, mu = 0) ...  ')
[Z, f_orim] = orimTheorem(A,M(1:n,1:n),0,eta2,k,objfun); fprintf('done.\n')

%% compute ORIM and ORIM_0 via Theorem 3.3 (P!=0 for different M)

fprintf('Computing ORIM using Theorem 3.3 (M_0 = [In, 0]) ...  ')
M_0 = eye(size(M));                                 % using M_0 = [I, 0]
[~,f_update0_P,~,~] = orimTheorem(A,M_0,P,eta2,k,objfun); fprintf('done.\n')

fprintf('Computing ORIM using Theorem 3.3 (M_1 = [In, mu]) ... ')
M_1 = eye(size(M)); M_1(:,end) = M(:,end);          % M_1 = [I, mu]
[~,f_update1_P,~,~] = orimTheorem(A,M_1,P,eta2,k,objfun); fprintf('done.\n')

fprintf('Computing ORIM using Theorem 3.3 (M_2 = [M, mu])  ... ')
M_2 = [M(1:n,1:n), zeros(n,1)];                     % M_2 = [M, 0]
[~,f_update2_P,~,~] = orimTheorem(A,M_2,P,eta2,k,objfun); fprintf('done.\n')

%% plot results

figure(1)

plot(f_TSVD,'s-', 'LineWidth',2,'color',[0.9290 0.6940 0.1250]), hold on,
plot(f_TTIK,'d-', 'LineWidth',2,'color',[0.4940 0.1840 0.5560])
plot(f_orim,'o-', 'LineWidth',2,'color',[0.4660 0.6740 0.1880])

plot(f_update0_P,'-',   'LineWidth',2,'color',[0.4940 0.1840 0.5560])
plot(f_update1_P,'--',  'LineWidth',2,'color',[0      0.4470 0.7410])
plot(f_update2_P,'m-.', 'LineWidth',2,'color',[0.4660 0.6740 0.1880])
plot(f,'k:','LineWidth',2)

fontSize = 12;
hLegend = legend('TSVD', 'TTik', 'ORIM$_0$', 'ORIM update, $\mathbf{M}_{(0)}$', 'ORIM update, $\mathbf{M}_{(1)}$', 'ORIM update, $\mathbf{M}_{(2)}$','ORIM update, $\mathbf{M}_{(3)}$');
set(hLegend,'FontSize',fontSize,'Interpreter','Latex','Location','northeastoutside');
xlabel('rank $r$','Interpreter','Latex','FontSize',fontSize)
ylabel('objective function $f(\mathbf{Z}_r)$','Interpreter','Latex','FontSize',fontSize)
set(gca,'FontSize',fontSize);
set(gca,'FontName','Times New Roman')
axis([1 50 9.2e5 1.28e6])  
box off