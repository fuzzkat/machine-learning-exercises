function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


[m, n] = size(X);

hypothesis = sigmoid( X * theta );

regSum = sum(zeroFirstTerm(theta).^2);
regularization = (lambda/(2*m)) * regSum;

costSum = sum( -y .* log(hypothesis) .- (1 .- y) .* log(1 .- hypothesis) );

J = (1/m) * costSum + regularization;



grad = (1/m) .* sum((repmat(hypothesis,1,n) .- repmat(y,1,n)) .* X);

% =============================================================

end
%!test;
%!  data = load('ex2data1.txt');
%!  X = data(:, [1, 2]); y = data(:, 3);
%!  XX = [ones(size(X,1),1),X];
%!  [m, n] = size(X);
%!  theta = [0.5;0.4;0.6]
%!  [cost,grad] = costFunctionReg(theta, XX, y, 0.6);
%!  sprintf('%1.3f', cost)
%!  sprintf('%1.3f', grad(1))
%!  sprintf('%1.3f', grad(2))
%!  sprintf('%1.3f', grad(3))

function T = zeroFirstTerm(theta)
  T = theta;
  T(1) = 0;
end
%!assert (zeroFirstTerm([5;4;3;2;1]), [0;4;3;2;1]);

function H = logisticHypothesis(X, theta)
  H = sigmoid( X * theta )
end
%!test;
%!  X = [1 2 3; 4 5 6; 7 8 9];
%!  theta = [0.2; 0.3; 0.4];
%!  assert (logisticHypothesis(X, theta), 1);

