function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i = 1:(size(X,1))
  idx(i) = findClosestCentroid(X(i,:), centroids, K);
end

% =============================================================

end

function cidx = findClosestCentroid(x, centroids, K)
  xarray = repmat(x,K,1);
  distances = xarray - centroids;
  norms = sqrt(sum(distances.^2,2));
  [val,cidx] = min(norms.^2);
end

%!test
%!  load('ex7data2.mat');
%!  initial_centroids = [3 3; 6 2; 8 5];
%!  idx = findClosestCentroids(X, initial_centroids);
%!  assert(idx(1:3), [1;3;2])


