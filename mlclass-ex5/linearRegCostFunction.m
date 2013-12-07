function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

costs = (X * theta) - y;
unRegJ = (1/(2*m)) * sum(costs .^2);

thetaNoBias = theta(2:end,:);
regTerm = (lambda/(2*m)) * sum(thetaNoBias .^2);

J = unRegJ + regTerm;

% =========================================================================
thetaZeroBias = [0;thetaNoBias];

for j = 1:size(theta,1)
  grad(j) = (1/m) * sum( (X * theta - y) .* X(:, j) ) + ((lambda/m) * thetaZeroBias(j));
endfor

grad = grad(:);
end

%!test
%!  X = [ones(10,1) sin(1:1.5:15)' cos(1:1.5:15)'];
%!  y = sin(1:3:30)';
%!  [J, grad] = linearRegCostFunction(X, y, [0.1 0.2 0.3]', 0.5);
%!  assert(J, 0.13856, 0.00001);

%!test
%!  load ('ex5data1.mat');
%!  m = size(X, 1);
%!  theta = [1 ; 1];
%!  [J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);
%!  assert(J, 303.993192, 0.000001);

%!test
%!  load ('ex5data1.mat');
%!  m = size(X, 1);
%!  theta = [1 ; 1];
%!  [J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);
%!  assert(grad(1), -15.303016, 0.000001);
%!  assert(grad(2), 598.250744, 0.000001);

%!test
%!  X = [ones(10,1) sin(1:1.5:15)' cos(1:1.5:15)'];
%!  y = sin(1:3:30)';
%!  [J, grad] = linearRegCostFunction(X, y, [0.1 0.2 0.3]', 0.5);
%!  assert(grad(1), 0.07071, 0.00001);
%!  assert(grad(2), 0.10090, 0.00001);
%!  assert(grad(3), 0.11923, 0.00001);

