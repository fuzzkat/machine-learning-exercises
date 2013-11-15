function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

[m, n] = size(X);

%XX = [ones(size(X,1),1),X];

hypothesis = sigmoid( X * theta );

J = 1/m * sum( -y .* log(hypothesis) .- (1 .- y) .* log(1 .- hypothesis) );

%for j = 1:length(theta)
%  grad(j) = 1/m * sum((hypothesis - y) .* X(:,j));
%endfor

grad = 1/m .* sum((repmat(hypothesis,1,n) .- repmat(y,1,n)) .* X);

% =============================================================

end

%!test;
%!  data = load('ex2data1.txt');
%!  X = data(:, [1, 2]); y = data(:, 3);
%!  XX = [ones(size(X,1),1),X];
%!  [m, n] = size(X);
%!  theta = zeros(n + 1, 1);
%!  [cost,grad] = costFunction(theta, XX, y);
%!  assert (sprintf('%1.3f', cost), '0.693');
%!  assert (sprintf('%1.0f', grad(1)), '-0');
%!  assert (sprintf('%1.0f', grad(2)), '-12');
%!  assert (sprintf('%1.0f', grad(3)), '-11');
%!  sprintf('%1.3f', cost)
%!  sprintf('%1.0f', grad(1))
%!  sprintf('%1.0f', grad(2))
%!  sprintf('%1.0f', grad(3))
