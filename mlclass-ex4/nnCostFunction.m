function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
a1 = X;
a1 = [ones(m, 1) a1];

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1), a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);
hypX = a3;

labelMatcher = repmat(1:num_labels,m,1); yMatcher = repmat(y,1,num_labels);
yMat = yMatcher == labelMatcher;

posDiff = -yMat .* log(hypX); negDiff = (1 - yMat) .* log(1 - hypX);
costMatrix = posDiff - negDiff;

sumOfSums = sum( sum( costMatrix ) );
unregularizedJ = 1/m * sumOfSums;

T1NoBias = Theta1(:,2:end);
T2NoBias = Theta2(:,2:end);

theta1Sum = sum( sum( T1NoBias .^ 2) );
theta2Sum = sum( sum( T2NoBias .^ 2) );
regularizationTerm = lambda / (2 * m) * (theta1Sum + theta2Sum);

J = unregularizedJ + regularizationTerm;

% -------------------------------------------------------------

for t = 1:m     % 1 to 5 for check
  % Step 1
  x_t = X(t,:);               % get one training instance
  y_t = y(t) == 1:num_labels;  % get correct result coerced into a binary vector

  a1_t = a1(t,:);  % 1x4 (inc. bias) - inputs straight from X
  a2_t = a2(t,:);  % 1x6 (inc. bias) - hidden layer activations
  z2_t = z2(t,:);  % 1x5
  a3_t = a3(t,:);  % 1x3             - results

  % Step 2 (for each output unit k in layer 3)
  delta3 = a3_t - y_t; % 1x3 (difference between activation and expected result)

  % Step 3
  gPrime = sigmoidGradient(z2_t);
  theta2Td3 = Theta2' * delta3'; % 3x6' * 1x3

  delta2 = theta2Td3(2:end) .* gPrime';  % should be 5x1 

  Theta1_grad += (delta2 * a1_t);  %4
  Theta2_grad += (delta3' * a2_t);
end

Theta1ZeroFirstCol = Theta1;
Theta1ZeroFirstCol(:,1) = 0;

Theta2ZeroFirstCol = Theta2;
Theta2ZeroFirstCol(:,1) = 0;

Theta1_grad = Theta1_grad * (1/m) + Theta1ZeroFirstCol * (lambda/m);
Theta2_grad = Theta2_grad * (1/m) + Theta2ZeroFirstCol * (lambda/m);

% =========================================================================



% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

