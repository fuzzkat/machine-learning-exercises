function [all_theta] = oneVsAll(X, y, num_labels, lambda)
m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);

X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================

  for c = 1:num_labels
    initial_theta = zeros(n + 1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
    all_theta(c, :) = theta(:);
  end

% =========================================================================


end
%!test;
%!  X = [];
%!  y = [1 0 0; 0 1 0; 0 0 1];
%!  num_labels = 8;
%!  lambda = [1,2,3,4];
%!  allTheta = oneVsAll(X, y, num_labels, lambda)

