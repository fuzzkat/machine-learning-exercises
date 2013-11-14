function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

prob = sigmoid(X * theta);
p = prob >= 0.5;

% =========================================================================


end

%!test;
%!  data = load('ex2data1.txt');
%!  X = data(:, [1, 2]); y = data(:, 3);
%!  XX = [ones(size(X,1),1),X];
%!  assert(sum(predict([-25.161272;0.206233;0.201470], XX)), 61)

