%% EXAMPLE 1 (Study 2) of [1] Paper_Ex1_2.m
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

%% set up problem generating matrix A, M, P, eta, and discretization
rng(0)                                           % set random number generator
n = 1000;                                        % dimension of problem m = n and p = n + 1
nkappa = 100;                                    % # of discretization points

fcn = @(n,d) heat(n,d);                          % set heat problem
kappa = linspace(1,2,nkappa);                    % choose discretization of kappa

M = speye(n); mu = zeros(n,1);                   % set matrix M with condition rank(M) = n
eta = 0.1; eta2 = eta^2;                         % set noise level parameter

param = {'tolRankUpdate',1e-5; 'display','off'}; % set algorithmic parameters

K1 = 3;                        % number of repeats for timing
K2 = 100;                      % number of repeats for solves

%% compare speed of SVD and ORIM as well as Tikhonov and ORIM reconstrution

% initialize, pre-allocate memory
timeTIK  = zeros(K1,nkappa); errTIK  = zeros(K2,nkappa);
timeORIM = zeros(K1,nkappa); errORIM = zeros(K2,nkappa);
P = 0;

for j = 1:nkappa
  
  fprintf('%d\n',j)
  
  % get matrix A,b,x for parameter d
  [A,bTrue,x] = fcn(n,kappa(j)); normx = norm(x);
  
  for k = 1:K1
    
    tic % timing ORIM
    [X,Y] = orim(A,mu,M,eta,P,param);
    timeORIM(k,j) = toc;
    
    tic % timing SVD
    [U,S,V] = svd(A);
    timeTIK(k,j) = toc;
    
  end
  
  ZTik = V*diag(diag(S)./(diag(S).^2+eta2))*U'; % create Tikhonov matrix
  if j == 1, P = ZTik; timeORIM(:,1) = timeTIK(:,1); else P = P+X*Y'; end % update P
  
  for k = 1:K2
    b = bTrue + eta*randn(n,1);          % add noise to vector bTrue
    errTIK(k,j)  = norm(ZTik*b-x)/normx; % report relative error
    errORIM(k,j) = norm(P*b-x)/normx;    % report relative error
  end
  
end

%% plot results on CPU timings and relative reconstruction errors

fontSize = 18;

figure(1), hold on
color = [0    0.4470    0.7410];
A75 = prctile(errORIM',75,2);
A25 = prctile(errORIM',25,2);
plot(kappa,median(errORIM,1),'Color',color,'LineWidth',2)
h = fill([kappa,fliplr(kappa)], [A25',fliplr(A75')],color); set(h,'facealpha',0.1,'EdgeColor','none')

figure(1), hold on
color = [0.8500    0.3250    0.0980];
A75 = prctile(errTIK',75,2);
A25 = prctile(errTIK',25,2);
plot(kappa,median(errTIK,1),':','Color',color,'LineWidth',2)
h = fill([kappa,fliplr(kappa)], [A25',fliplr(A75')],color); set(h,'facealpha',0.1,'EdgeColor','none')

legend('median ORIM','25-75 percentile','median Tikhonov','25-75 percentile');
xlabel('$\kappa$','Interpreter','LaTeX','FontSize',fontSize)
ylabel('relative error','Interpreter','LaTeX','FontSize',fontSize)
set(gca,'FontSize',fontSize);
set(gca,'FontName','Times New Roman')

fontSize = 16;
figure(2), hold on
color = [0    0.4470    0.7410];
A75 = prctile(timeORIM',75,2);
A25 = prctile(timeORIM',25,2);
plot(kappa,median(timeORIM,1),'Color',color,'LineWidth',2)
h = fill([kappa,fliplr(kappa)], [A25',fliplr(A75')],color); set(h,'facealpha',0.1,'EdgeColor','none')

figure(2), hold on
color = [0.8500    0.3250    0.0980];
A75 = prctile(timeTIK',75,2);
A25 = prctile(timeTIK',25,2);
plot(kappa,median(timeTIK,1),':','Color',color,'LineWidth',2)
h = fill([kappa,fliplr(kappa)], [A25',fliplr(A75')],color); set(h,'facealpha',0.1,'EdgeColor','none')

hLegend = legend('median ORIM','25-75 percentile','median Tikhonov','25-75 percentile');
set(hLegend,'FontSize',fontSize-2);
xlabel('$\kappa$','Interpreter','LaTeX','FontSize',fontSize)
ylabel('CPU time [s]','Interpreter','LaTeX','FontSize',fontSize)
set(gca,'FontSize',fontSize);
set(gca,'FontName','Times New Roman')
