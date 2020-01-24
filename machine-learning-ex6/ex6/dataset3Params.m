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

C_m = [0.01;0.03; 0.1; 0.3; 1;3;10;30];
sigma_m = C_m;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model, Xval);
cost = mean(double(predictions ~= yval));

    for i = 1:length(C_m)
        for j = 1 : length(sigma_m),
                       model = svmTrain(X,y,C_m(i),@(x1,x2)gaussianKernel(x1,x2,sigma_m(j)));
                       predictions = svmPredict(model,Xval);
                       tmp_cost = mean(double(predictions ~= yval));
                       if (tmp_cost < cost)
                           cost = tmp_cost;
                           c = C_m(i);
                           sigma = sigma_m(j);
                        endif
        end                
    end       
%   model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
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







% =========================================================================

end
