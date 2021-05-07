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
    %       of the cost function (computeCost) and gradient here.\

    h = theta' * X';
    x1 = X(:,1);
    x2 = X(:,2);
    temp0 = 0;
    temp1 = 0;
    sum1=0;
    sum2=0;


    for i = 1:m
        
        sum1 = sum1 + (alpha/m*(h(i)-y(i))*x1(i));
        sum2 = sum2 + (alpha/m*(h(i)-y(i))*x2(i));
    end
    

        temp0 =  theta(1) -sum1;      
        temp1 = theta(2) - sum2;
   
        theta(1)= temp0;
        theta(2) = temp1;
    

    %end

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
 
   
    

end

end
