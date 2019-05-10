function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% 前向传播计算h(x)
%~ % 添加第一层偏置单元
%~ X_1 = [ones(m, 1) X];
%~ % 进行隐藏层神经单元的计算
%~ X_2 = sigmoid(X_1 * Theta1');

%~ % 添加第二层偏置单元
%~ X_2 = [ones(m, 1) X_2];
%~ % 进行输出层神经单元的计算
%~ h_x = sigmoid(X_2 * Theta2');

%~ % 统计预测值
%~ [x, ix] = max(h_x, [], 2);
%~ p = ix;

[x, ix] = max(sigmoid([ones(m, 1) sigmoid([ones(m, 1) X] * Theta1')] * Theta2'), [] ,2);
p = ix;

% =========================================================================


end
