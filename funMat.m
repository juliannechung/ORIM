classdef funMat
  %funMat class
  %  A funMat object is used to represent a matrix A,
  %       where A is accessed via function evaluations for both
  %       matrix-vector and matrix-transpose-vector multiplications
  %
  %  The funMat class has inputs:
  %     Afun  - matvec function handle
  %     Atfun - transpose function handle
  %   and is based on a structure with three fields:
  %     Afun
  %     Atfun
  %     transpose - indicates if the matrix has been transposed
  %
  %  Calling Syntax:
  %
  %    P = funMat    (returns object with empty fields)
  %    P = funMat(funMat) (returns funMat)
  %    P = funMat(Afun, Atfun)
  %
  % J. Chung, 3/4/2016
  
  properties
    Afun
    Atfun
    transpose
  end
  
  methods
    function P = funMat(varargin) % Constructor
      switch nargin
        case 0
          P.transpose = false;
          P.Afun = [];
          P.Atfun = [];
        case 1
          if isa(varargin{1}, 'funMat')
            P = varargin{1};
          else
            error('Incorrect input arguments')
          end
        otherwise
          P.transpose = false;
          if nargin == 2
            P.Afun = varargin{1};
            P.Atfun = varargin{2};
          else
            error('Incorrect number of input arguments')
          end
      end
    end
    
    function P = ctranspose(P) % Overload transpose
      P.transpose = not(P.transpose); % switches booleen transpose flag
    end
    
    function y = mtimes(arg1, arg2) % Overload matrix vector multiplicaiton
      %   Implement B*s and B'*s for funMat object B and "vector" s.
      
      if isa(arg1,'funMat')
        % check to see of arg2 is a scalar
        if isa(arg2,'double') && length(arg2) == 1
          error('Matrix times a scalar not implemented yet')
        else
          if numel(arg2) ~= length(arg2) %  Check to see if arg2 is not a vector
            warning('Warning:funMat/mtimes assumes input is a vector')
          end
          
          if arg1.transpose 
            % Matrix transpose times vector
            y = arg1.Atfun(arg2);
          else
            % Matrix times vector
            y = arg1.Afun(arg2);
          end
        end
        
      elseif (( isa(arg1, 'double')) && (length(arg1)==1))
        error('Multiplication is not implemented.')
      else
        error('Multiplication is not implemented.')
      end
    end
        
  end % methods
end % classdef

