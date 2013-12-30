function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
errors = ((X * Theta') - Y) .* R;
squared_errors = errors.^2;
J = 0.5 * sum(sum(squared_errors));

X_grad = errors * Theta;
Theta_grad = errors' * X;

% Regularize
J = J + (lambda/2) * sum(sum(Theta.^2)) + (lambda/2) * sum(sum(X.^2));
X_grad = X_grad + lambda * X;
Theta_grad = Theta_grad + lambda * Theta;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end

%!test
%!  load ('ex8_movies.mat');
%!  load ('ex8_movieParams.mat');
%!  num_users = 4; num_movies = 5; num_features = 3;
%!  X = X(1:num_movies, 1:num_features);
%!  Theta = Theta(1:num_users, 1:num_features);
%!  Y = Y(1:num_movies, 1:num_users);
%!  R = R(1:num_movies, 1:num_users);
%!  J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, num_features, 0);
%!  assert(J, 22.22, 0.01)

%!test
%!  lambda = 0;
%!  X_t = rand(4, 3);
%!  Theta_t = rand(5, 3);
%!  Y = X_t * Theta_t';
%!  Y(rand(size(Y)) > 0.5) = 0;
%!  R = zeros(size(Y));
%!  R(Y ~= 0) = 1;
%!  X = randn(size(X_t));
%!  Theta = randn(size(Theta_t));
%!  num_users = size(Y, 2);
%!  num_movies = size(Y, 1);
%!  num_features = size(Theta_t, 2);
%!  numgrad = computeNumericalGradient( @(t) cofiCostFunc(t, Y, R, num_users, num_movies, num_features, lambda), [X(:); Theta(:)]);
%!  [cost, grad] = cofiCostFunc([X(:); Theta(:)],  Y, R, num_users, num_movies, num_features, lambda);
%!  diff = norm(numgrad-grad)/norm(numgrad+grad);
%  disp([numgrad grad]);
%!  assert(diff, 0, 1e-9);


