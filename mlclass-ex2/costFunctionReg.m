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

regTerm = zeroFirstTerm(theta).^2;
regularization = (lambda/(2*m)) * sum(regTerm);

cost = -y .* log(hypothesis) .- (1 .- y) .* log(1 .- hypothesis);

J = (1/m) * sum(cost) + regularization;

gradnorm = (lambda/m) .* zeroFirstTerm(theta);
gradsum = sum((repmat(hypothesis,1,n) .- repmat(y,1,n)) .* X);
grad = (1/m) .* gradsum' + gradnorm;

% =============================================================

end
%!test;
%!  data = load('ex2data1.txt');
%!  X = data(:, [1, 2]); y = data(:, 3);
%!  X = mapFeature(X(:,1), X(:,2));
%!  theta = zeros(size(X, 2), 1);
%!  [cost,grad] = costFunctionReg(theta, X, y, 1);



function T = zeroFirstTerm(theta)
  T = theta;
  T(1) = 0;
end
%!test;
%!  assert (zeroFirstTerm([5;4;3;2;1]), [0;4;3;2;1]);

