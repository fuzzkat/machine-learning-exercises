function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

g = 1./(1 .+ e.^-z);

% =============================================================

end

%!assert (sigmoid(0), 0.5);
%!assert (sigmoid(5) > 0.5);
%!assert (sigmoid(-5) < 0.5);

%!test s = sigmoid([0,5,-5])
%!  assert (s(1), 0.5);
%!  assert (s(2) > 0.5);
%!  assert (s(3) < 0.5);

