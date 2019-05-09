function mainMulti(i)

%多元线性回归

cd 'C:\Users\wangk\Desktop\machine-learning\machine-learning-ex1\ex1'
load ex1data2.txt;

%X设计矩阵 y实际值
X = [ones(length(ex1data2), 1) ex1data2(:, 1) ex1data2(:, 2)];
y = ex1data2(:, 3);

if i == 1,
	printf('Gradient Descent Multi\n');
	%特征规范化
	[X_norm, mu, sigma] = featureNormalize(X);
	%~ X_norm, mu, sigma
	%梯度下降
	theta = zeros(size(X_norm, 2), 1);
	alpha = 0.001;
	num_iters = 10000;
	
	[theta, J_history] = gradientDescentMulti(X_norm, y, theta, alpha, num_iters);
	
	theta
	X_pre = ([1 3000 5]-mu)./sigma;
	X_pre(:, 1) = 1;
	printf("size:3000 room:5 -> prediction-price: %.2f\n", X_pre*theta);
	
elseif i == 2,
	printf('Normal Eqn\n');
	theta = normalEqn(X, y);
    printf("size:3000 room:5 -> prediction-price: %.2f\n", [1 3000 5]*theta);

else
	printf('Error!!!\n');
	
end


end
