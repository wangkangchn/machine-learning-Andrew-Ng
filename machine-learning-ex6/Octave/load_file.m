function vocabList = getVocabList()
%GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
%cell array of the words
%   vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt 
%   and returns a cell array of the words in vocabList.

% 返回一个由vocabulary中单词组成的数组

%% Read the fixed vocabulary list
fid = fopen('vocab.txt')

% Store all dictionary words in cell array vocab{}
n = 1899;  % Total number of words in the dictionary

% For ease of implementation, we use a struct to map the strings => integers
% In practice, you'll want to use some form of hashmap
vocabList = cell(n, 1);		# 创建一个n*1的单元数组
for i = 1:n
    % Word Index (can ignore since it will be = i)
    fscanf(fid, '%d', 1);		% 读取下标抛弃
    % Actual Word
    vocabList{i} = fscanf(fid, '%s', 1);	% 读取单词
end
%~ vocabList
fclose(fid);

end
