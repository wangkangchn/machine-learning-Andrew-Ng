function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% 样本与聚类中心距离
distance = zeros(K, 1);

% 循环计算每一个样本与聚类中心的距离
for i=1:size(X,1)
	distance = sum(((X(i, :) - centroids) .^ 2), 2);
	% 提取最小距离的下标
	[value, idx(i)] = min(distance, [], 1);
end	

%~ % 样本与聚类中心距离
%~ distance = zeros(size(X, 1), K);
%~ % 遍历每一个中心, 进行计算
%~ for k=1:K
	%~ k
	%~ % 对角线上元素是我们所需要的
	%~ distance(:, k) = sum(eye(size(X, 1)) .* (((X - centroids(k, :)) * (X - centroids(k, :))')), 2);
%~ end	
%~ [value, idx] = min(distance, [], 2);
% =============================================================

end

