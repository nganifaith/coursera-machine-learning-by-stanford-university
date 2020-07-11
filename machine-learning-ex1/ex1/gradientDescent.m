function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % for a x(0) = 1
    % h(X) = theta(0) + theta(1)X. We are dealing with 1 index meaning
    % h(x) = theta(1) + theta(2)X
 
    x = X(:, 2);
    hypotesis = theta(1) + (theta(2)*x);

    theta_zero = theta(1) - alpha * (1/m) * sum(hypotesis - y);
    theta_one = theta(2) - alpha * (1/m) * sum((hypotesis - y).*x);
    
    % new values for theta as a row vector
    theta = [theta_zero; theta_one]

     
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
