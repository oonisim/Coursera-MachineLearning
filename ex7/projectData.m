function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

% Backup on 16 DEC 2016
%for i = 1:size(X, 1)
%    Z(i, :) = X(i, :) * U(:, 1:K); 
%end

%----------------------------------------------------------------------
% [Data Set X]           : Row is a measurement. Column is a feature.
% [Eigen Vector Matrix U]: Each column is an eigen vector. 
% UK: Primary K components of U is U(:, 1:K)
%----------------------------------------------------------------------
% Projection from X data space to primary eigen vector space Z is X * UK.
%----------------------------------------------------------------------
UK = U(:, 1:K);
Z = X * UK;

% =============================================================

end
