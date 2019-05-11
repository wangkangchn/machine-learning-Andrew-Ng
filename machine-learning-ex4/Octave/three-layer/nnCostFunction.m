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

% 对标记y重新编码 变为 5000*10 矩阵
% *********注意映射的时候, 0便为10, 相对应的输出单元是最后一个!!!为了与下标匹配
y_recode = zeros(m, num_labels);
for i=1:num_labels
	y_recode(y==i, i) = 1;
end

% Step 1 前向传播计算h(x)
a_1 = [ones(m, 1) X];				% 5000*401
z_2 = a_1 * Theta1';				% 5000*25
a_2 = [ones(m, 1) sigmoid(z_2)];	% 5000*26
z_3 = a_2 * Theta2';				% 5000*10
h_x = sigmoid(z_3);					% 5000*10

% Step 2 代价函数
J = - 1 / m * sum(sum(eye(m) .* (y_recode * log(h_x')  ...
		+ (1 - y_recode) * log(1 - h_x')))) ...
		+ lambda / (2 * m) * (sum(sum(Theta1(:, 2:end) .^ 2)) ...
		+ sum(sum(Theta2(:, 2:end) .^ 2)));

% Step 3 反向传播 之 计算输出层z的偏导数
delta_3 = h_x - y_recode;	 		% 5000*10 Theta1 25*401 Theta2 10*26

% Step 4 反向传播 之 计算对隐藏层输入z的偏导数, 不考虑偏置单元的影响
delta_2 = (delta_3 * Theta2)(:, 2:end) .* sigmoidGradient(z_2);	% 5000*25

% Step 5 反向传播 之 计算参数的偏导数	
Theta2_grad += (delta_3' * a_2) / m;	% 10*26
Theta1_grad += (delta_2' * a_1) / m;	% 25*401

% 正则化
Theta2_grad(:, 2:end) += lambda / m * Theta2(:, 2:end);
Theta1_grad(:, 2:end) += lambda / m * Theta1(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
