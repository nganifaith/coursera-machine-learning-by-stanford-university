function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hypothesis  = X * theta;
len_theta = length(theta)
regularization_var = (lambda/(2*m))*(sum(theta(2:len_theta).^2));


J = (1/(2*m)) * sum((hypothesis - y).^2) + regularization_var


linear_reg = (1/m)*( X'*(hypothesis - y));
regularized_grad = linear_reg + (lambda/m)*theta;

grad = [linear_reg(1); regularized_grad(2: len_theta)]







% =========================================================================

grad = grad(:);

end
