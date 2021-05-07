function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %
h= theta' *X' ;
sum0=0;
sum1=0;
sum2=0;
x0= X(:,1);
x1=X(:,2);
x2=X(:,3);
temp0 = 0;
temp1 = 0;
temp2 = 0;

for i = 1:m
    sum0 = sum0 + alpha/m*(h(i)-y(i))*x0(i);
    sum1 = sum1 + alpha/m*(h(i)-y(i))* x1(i);
    sum2 = sum2 + alpha/m*(h(i)-y(i))*x2(i);
    disp(sum0)
 
end

temp0 = theta(1) - sum0;
%temp1 = theta(2) - sum1;
%temp2 = theta(3) - sum2;

theta(1) = temp0;
%theta(2) = temp1;
%theta(3) = temp2;








    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
