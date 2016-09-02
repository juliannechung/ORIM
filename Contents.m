%% Contents.m
%   This folder contains MATLAB code to accompany the paper:
% 
%   Julianne Chung and Matthias Chung. Optimal Regularized Inverse
%     Matrices for Inverse Problems, preprint, 2016.
%
% Authors:
%   (c) Julianne Chung (e-mail: jmchung@vt.edu)            in March 2016
%       Matthias Chung (e-mail: mcchung@vt.edu)
%       
% Versions:
%   1.0 Initial release [March 2016]
%   2.0 Small bugfixes and modifications to improve stability and 
%       convergence of method [September 2016]
%
% These codes require the following packages:
%         Regularization Tools package: Hansen. Regularization tools: A
%             package for analysis and solution of discrete ill-posed 
%             problems. Numerical Algorithms, 1994.
%
%         RestoreTools package: Nagy, Palmer, and Perrone. Iterative 
%             Methods for image deblurring: A Matlab object oriented 
%             approach. Numerical Algorithms, 2004.


%% Script Files used to generate results in the paper
%
%  Paper_Ex1_1.m        This script compares function values for TSVD,
%                         TTik, ORIM_0 and ORIM.
%                         Results correspond to Experiment 1.
%
%  Paper_Ex1_2.m        This script compares timings and reconstruction
%                         errors for ORIM vs Tikhonov.
%                         Results correspond to Experiment 1.
%
%  Paper_Ex2_Deblur.m   This script computes reconstructions for an image 
%                         deblurring problem. We compare Tikhonov 
%                         regularization and ORIM updates to Tikhonov.
%                         Results correspond to Experiment 2.
%
%% Supporting MATLAB functions included here
%
% orim.m              computes an ORIM using the rank-update approach
%                       described in the paper
%
% orim_Theorem.m      computes an ORIM using the theorem from the paper
%                       (not efficient for large-scale problems)
%
% funMat.m            a matlab class that is used to create objects that 
%                       represent a matrix A, where A is accessed via 
%                       function evaluations for matrix-vector and 
%                       matrix-transpose-vector multiplications
%
% OptTikTol.m        function used to compute the optimal Tikhonov
%                       regularization parameter (assumes the true signal 
%                       is known and the SVD is available)
%

