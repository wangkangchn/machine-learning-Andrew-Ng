function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%	使用不同的训练集计算误差
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------



%~ for i = 1:m
	%~ % Compute train/cross validation errors using training examples 
	%~ % X(1:i, :) and y(1:	i), storing the result in 
	%~ % error_train(i) and error_val(i)
	
	%~ % 使用不用的训练样本进行参数学习 -> 进行验证
	%~ [theta] = trainLinearReg(X(1:i, :), y(1:i), lambda);
	%~ % 使用不同训练样本计算训练误差, lambda=0
	%~ error_train(i) = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
	%~ % 使用全部交叉验证样本计算交叉验证误差, lambda=0
	%~ error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);

%~ end

for i = 1:m

	% 随机选择训练样本以及验证样本, 重复50次取误差的平均值
	for t=1:50
		sel_1 = randperm(size(X, 1));		% 随机选取行下标
		sel_2 = randperm(size(Xval, 1));
		sel_1 = sel_1(1:i);
		sel_2 = sel_2(1:i);
		
		% 使用不用的训练样本进行参数学习 -> 进行验证
		[theta] = trainLinearReg(X(sel_1, :), y(sel_1), lambda);
		% 使用不同训练样本计算训练误差, lambda=0
		error_train(i) += linearRegCostFunction(X(sel_1, :), y(sel_1), theta, 0);
		% 使用全部交叉验证样本计算交叉验证误差, lambda=0
		error_val(i) += linearRegCostFunction(Xval(sel_2, :), yval(sel_2), theta, 0);
	end
	
end
error_train /= 50;
error_val /= 50;

% -------------------------------------------------------------

% =========================================================================

end
