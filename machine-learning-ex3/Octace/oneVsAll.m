function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);				%n+1 偏置

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

% Set Initial theta
initial_theta = zeros(n + 1, 1);

% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 50);

% 每个循环训练一个分类器
for c = 1:num_labels
	printf("%dth classifier\n", c)	
	% Run fmincg to obtain the optimal theta
	% This function will return theta and the cost 
	
	% 对 y == c 的说明, y中保存的是所有训练样本的标记1-10, y==c 当y中的标记
	% 与c相等时返回1, 不等返回0, 这样每次训练就可以只对特定的标记进行训练
	% 对对对这跟我之前想的是一模一样, 开心
	[theta] = ...
	fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
		initial_theta, options);
	
	% 保存每一个分类器的参数theta
	all_theta(c, :) = reshape(theta, 1, size(all_theta)(2));
end

printf('\nComplete the training.\n\n')
% =========================================================================


end
