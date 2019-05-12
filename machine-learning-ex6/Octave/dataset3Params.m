function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% 可供选择C, sigma的值
value = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
% 保存每一组C, sigma的误差
error = zeros(length(value), length(value));
% 选择C
for c=1:length(value)	% 行
	% 选择sigma
	for s=1:length(value)	% 列
		%~ fprintf('C: %f, sigma: %f', value(c), value(s));
		% 使用C sigma训练模型
		model= svmTrain(X, y, value(c), @(x1, x2) gaussianKernel(x1, x2, value(s))); 
		% 进行预测
		predictions = svmPredict(model, Xval);
		% 计算误差
		
		error(c, s) = mean(double(predictions ~= yval));
		fprintf('error: %f\n\n', error(c, s))

	end
end

% 提取最小的误差
[dummy, p] = min(min(error, [], 1), [], 2);
% 查找微小误差的位置
[row col] = find(error==dummy);
C = value(row);
sigma = value(col);
fprintf('min C: %f, min sigma: %f, min error = %f\n', C, sigma, dummy)

% =========================================================================

end
