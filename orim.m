function [X,Y,f,F] = orim(A,mu,M,eta,P,param)
%
% function [X,Y,f,F] = orim(A,mu,M,eta,P,param)
%
% Authors:
%   (c) Julianne Chung (e-mail: jmchung@vt.edu)            in February 2016
%       Matthias Chung (e-mail: mcchung@vt.edu)
%
% MATLAB Version: 8.6.0.267246 (R2015b)
%
% Description:
%   Solves min_Z f(Z) = ||((Z+P)A-I)M||_F^2 + eta^2*||Z+P||_F^2
%   where Z = XY' with X, Y of maximal rank rMax. Matrix Z is a regularized
%   update of inverse approximation P.
%
% Input arguments:
%   A                    - objective matrix [m x n]
%   mu                   - mean estimate of desired solution x [n x 1]
%   M                    - estimate of square root of covariance matrix of
%                          desired solution M [n x n]
%   eta                  - noise regularization parameter (> 0)
%   P                    - initial inverse approximation matrix [n x m]
%   #param               - further options of algorithm
%     AM                 - direct input of AM as matrix or function handle
%                          (if selected ignoring input A)
%     display            - print to display [ 'off' | 'iter' | {'rank'} | 'final' ]
%     rMax               - maximum number of ranks (default floor(m*n/(m+n)))
%     maxIterAlternating - maximal number of iterations [ ceil(min(m,n)/100) ]
%     tolAlternating     - tolerance for stopping alternating directions  [ 1e-6 ]
%     tolIterSolver      - tolerance for MATLAB's iterative LSQR solver   [ 1e-6 ]
%     tolRankUpdate      - tolerance for stopping criteria of rank update [ 1e-6 ]
%     X                  - initially known low rank approximation X on A [n x k]
%     Y                  - initially known low rank approximation Y on A [m x k]
%     fprev              - fprev = 0 will precomute function value
%                          otherwise function value is set to fprev (default fprev = 0)
%
% Output arguments:
%   X                    - low rank matrix X [n x r]
%   Y                    - low rank matrix Y [m x r]
%   f                    - misfit at final rank
%   F                    - misfit at each iteration
%
% References:
%    [1] Julianne Chung and Matthias Chung. Optimal Regularized Inverse
%        Matrices for Inverse Problems, preprint, 2016.
%

% set default parameters
tolAlternating     = 1e-6;               % relative tolerance for alternating between x and y
tolIterSolver      = 1e-6;               % tolerance of iterative LSQR solver on y
tolRankUpdate      = 1e-6;               % tolerance on improvement between ranks
[m,n]              = size(A);            % get size of problem
maxIterAlternating = ceil(min(m,n)/100); % default maximal alternating directions
r                  = 0;                  % default start without rank approximation
display            = 'rank';             % default display in command window
rMax               = floor(m*n/(m+n));   % max rank where decomposition
fprev              = 0;                  % if fprev = 0 function value will be precomputed otherwise fprev will be taken as initial function value

% rewrite default parameters if selected
if nargin == nargin(mfilename)
  for j = 1:size(param,1), eval([param{j,1},'= param{j,2};']); end
end

% display and algorithm info
if strcmp(display, 'iter') || strcmp(display,'rank') || strcmp(display,'final')
  fprintf('\nComputing ORIM with Rank Update (c) Julianne Chung & Matthias Chung 2/2016\n');
end

% check maximal rank, desired rank should be smaller than matrix columns and rows
if rMax > min(m,n)
  rMax = min(m,n);
  warning(['rMax to large adjusting rank r = min(m,n) =  ',num2str(rMax)])
end

% check if an initial low rank approximation Z = XY' is already provided
if exist('X','var') && exist('Y','var')
  if size(X,2) == size(Y,2)
    r = size(X,2); % adjust the starting rank
    X = [X, zeros(n,rMax-r)]; Y = [Y, zeros(m,rMax-r)]; % pre-allocate space and update X and Y
    warning(['Initial low rank matrices X and Y found. Adjusting initial rank r = ',num2str(r), '.'])
  else
    error('Low rank approximations X and Y must have same number of columns.')
  end
else
  X = zeros(n,rMax); Y = zeros(m,rMax);  % pre-allocate memory to save X and Y
end

if nargout > 3, F = zeros(rMax,1); end   % record objective function value

% display and algorithm info
if strcmp(display, 'iter') || strcmp(display,'rank')
  fprintf('Pre-computing matrices ...')
end

% initialize method
if ~exist('AM','var')
  M       = [M,mu];                                      % combine mu and M into one matrix
  AM      = A*M;                                         % combine A and M
end
eta2    = eta^2;                                         % pre-compute the square of eta
objectFcn = @(y,flag) mviFcn(AM,y,eta,n,flag);           % define operation [AM';eta I]*y and [AM';eta2 I]'*b
if ~fprev
  fprev = norm(P*AM - M,'fro')^2 + eta2*norm(P,'fro')^2; % precomputing function value for X*Y' = 0
end
clear A mu                                               % clear unused variables

% display and algorithm info
if strcmp(display, 'iter') || strcmp(display,'rank')
  fprintf(' done.\n')
end

while 1
  
  r = r+1; % increase rank
  
  % display and algorithm info
  if strcmp(display, 'iter') || strcmp(display,'rank')
    fprintf('Rank %d ...',r), tic
    if strcmp(display, 'iter'), fprintf('\n'), end
  end
  
  % initialize alternating directions
  iter = 1; fOld = inf; xOld = inf; yOld = inf;
  
  y = ones(m,1); % initial guess of y
  
  % pre-compute constants and vectors for alternating directions
  yAM = (AM'*y)';  %yAM = y'*AM;
  AMMAyeta2y = AM*yAM' + eta2*y;
  yAMMAyeta2yy = y'*AMMAyeta2y;
  
  while 1 % alternating directions over x and y
    
    % compute optimal x
    x = (M*yAM' - P*AMMAyeta2y - X(:,1:r-1)*(Y(:,1:r-1)'*AMMAyeta2y))/yAMMAyeta2yy;
    
    % normalize x
    x = x/norm(x);
    
    % pre-compute M'*x and (P' + Y*X')*x
      Mx = M'*x; PpXYx = P'*x + Y(:,1:r-1)*(X(:,1:r-1)'*x);
    
    % compute optimal y
    [y, ~, ~] = lsqr(objectFcn, [Mx - AM'*PpXYx; -eta*PpXYx], tolIterSolver); 
    
    % pre-computes for optimal x in next iteration
    yAM = (AM'*y)';
    AMMAyeta2y = AM*yAM' + eta2*y;
    yAMMAyeta2yy = y'*AMMAyeta2y;
    
    % compute update on objective function
    f = yAMMAyeta2yy + 2*(AMMAyeta2y'*PpXYx) - 2*(yAM*Mx) + fprev;
    
    % check stopping criteria for alternating directions
    STOP1 = abs(f - fOld) < tolAlternating*f;                                   % relative improvement in f
    STOP2 = norm(xOld - x,'inf') <= sqrt(tolAlternating) * (1 + norm(x,'inf')); % relative improvement in x % no need to compute norm since ||x|| = 1
    STOP3 = norm(yOld - y,'inf') <= sqrt(tolAlternating) * (1 + norm(y,'inf')); % relative improvement in y
    STOP4 = iter >= maxIterAlternating;                                         % max number of alternating optimizations
    
    % display and algorithm info
    if strcmp(display,'iter')
      fprintf('%5d %14.6e %14.6e %4d%1d%1d \n',...
        iter, f, abs(f - fOld)/f, STOP1,STOP2,STOP3,STOP4);
    end
    
    % execute stopping criteria for alternating directions
    if STOP1 || STOP2 || STOP3 || STOP4
      break
    end
    
    fOld = f; xOld = x; yOld = y; % keep old values for stopping criteria
    iter = iter + 1;              % increase counter
    
  end % rank 1 alternating directions over x and y
  
  % display and algorithm info
  if strcmp(display, 'iter') || strcmp(display,'rank')
    fprintf(' diff(f_r) =  %e ... [%f s] [%d] done.\n',fprev - f,toc,iter)
  end
  
  % define stopping criteria for the rank
  STOP1 = abs(fprev - f) < tolRankUpdate*abs(f);
  STOP2 = r > rMax-1;
  
  if nargout > 3, F(r) = f; end % record objective function value
  
  fprev = f; % update function value
  
  % saving optimal X and Y
  X(:,r) = x; Y(:,r) = y;
  
  % execute stopping criteria for the rank
  if STOP1 || STOP2
    if STOP2 && ~STOP1
      warning('Matlab:orim:maxRank',...
        'Maximal rank reached. Improving on ORIM matrix still possible.')
    end
    if (strcmp(display, 'iter') || strcmp(display,'rank')) &&  (STOP1 && ~STOP2)
      fprintf('Rank improvement tolerance %1.2e reached at rank = %d.\n',tolRankUpdate,r)
    end
    break
  end
  
end

X = X(:,1:r); Y = Y(:,1:r);      % reduce X and Y if not all ranks used

if nargout > 3, F = F(1:r); end  % record objective function value

end

function y = mviFcn(AM,b,eta,n,flag)
% This function performs
%      [  AM' ]
% y =  [ eta*I] * b
% and transform times b
% This is an approriate function handle for MATLAB's iterative LSQR solver

if strcmp(flag,'notransp') % use operator
  y = [AM'*b; eta*b];
else % case using the transposed operator
  y = AM*b(1:n+1) + eta*b(n+2:end);
end

end