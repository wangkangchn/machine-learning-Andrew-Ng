function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;			% 将返回的变量先声明出, 随后赋值返回

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

predictions = X * theta;			% 预测
Errors = predictions - y;			% 误差平方

% 代价值     X'*X 平方和
J = 1 / (2 * m) * (Errors' * Errors);

% =========================================================================

end
