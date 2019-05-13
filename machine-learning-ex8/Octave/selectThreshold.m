function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

	% 选择阈值epsilon, 计算查准率以及召回率, 计算F分
	% 先使用pval进行预测, 得到预测的数据后使用yval来计算tp, fp, fn
	
	% 依据当前 epsilon 进行预测, 小于epsilon为异常点标记为1
	cvPredictions = pval < epsilon;
	
	% 计算tp, 预测以及实际值均为1的个数
	tp = sum((cvPredictions == 1) & (yval == 1));
	
	% 计算fp, 预测为1实际为0的个数
	fp = sum((cvPredictions == 1) & (yval == 0));
	
	% 计算fn, 预测为0实际为1的个数
	fn = sum((cvPredictions == 0) & (yval == 1));
	
	% 计算查准率
	prec = tp / (tp + fp);
	
	% 计算召回率
	rec = tp / (tp + fn);
	
	% 计算F分
	F1 = 2 * prec * rec / (prec + rec);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
