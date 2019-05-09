function g = sigmoid(z)
g = zeros(size(z));

g = 1 ./ (1 + exp(-z));			%取倒数要用./	对每一个元素进行计算

end
