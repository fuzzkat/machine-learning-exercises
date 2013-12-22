function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.

for k = 1:K
  XforK = X(idx==k,:);
  centroids(k,:) = (1/size(XforK,1)) * sum(XforK);
end

% =============================================================


end


%!test
%!  load('ex7data2.mat');
%!  K = 3; % 3 Centroids
%!  initial_centroids = [3 3; 6 2; 8 5];
%!  idx = findClosestCentroids(X, initial_centroids);
%!  centroids = computeCentroids(X, idx, K);
%!  assert(centroids(1,:), [ 2.428301 3.157924 ], 0.000001)
%!  assert(centroids(2,:), [ 5.813503 2.633656 ], 0.000001)
%!  assert(centroids(3,:), [ 7.119387 3.616684 ], 0.000001)

