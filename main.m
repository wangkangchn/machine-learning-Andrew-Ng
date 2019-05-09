function main(i)
cd 'C:\Users\wangk\Desktop\machine-learning\machine-learning-ex1\ex1'
load ex1data1.txt;

%X设计矩阵 y实际值
X = [ones(length(ex1data1), 1) ex1data1(:, 1)];
y = ex1data1(:, 2);

if i == 1,
	printf('Gradient Descent\n');
	theta = zeros(size(X, 2), 1);
	alpha = 0.01;
	
	[theta, J] = gradientDescent(X, y, theta, alpha, 5000);
	
	%~ plotData(ex1data1(:, 1), y);
	%~ hold on, plot(ex1data1(:, 1), X*theta, 'g');
	
	theta
elseif i == 2,
	printf('Normal Eqn\n');
	[theta] = normalEqn(X, y);

else
	printf('Error!!!\n');
end
end
