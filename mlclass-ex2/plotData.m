function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

positive = find(y==1);
negative = find(y==0);

plot(X(positive,1), X(positive,2), 'k+;Admitted;', 'MarkerSize', 6);
plot(X(negative,1), X(negative,2), 'ko;Not Admitted;', 'MarkerFaceColor', 'r', 'MarkerSize', 6, 'LineWidth', 2);

%xlabel('Exam 1 score')
%ylabel('Exam 2 score')
legend('boxon')

% =========================================================================



hold off;

end
