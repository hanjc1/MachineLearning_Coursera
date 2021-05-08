function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(theta'*X');
temp0 = 0;

sum0=0;


for i = 1:m
    temp0 = temp0 + [(-1)*y(i)*log(h(i))-(1-y(i))*log(1-h(i))];
end

for j = 1:length(theta)
    if j ~= 1
    sum0 = sum0 + theta(j)^2;
    end
    
  
end


J = 1/m*temp0 + lambda/(2*m)*sum0;


for j = 1:length(theta)
    temp1=0;
    temp2=0;
    
    for i = 1:m
        
        if j == 1
        temp1 = temp1 + (h(i)-y(i))*X(i,j);
        grad(j) = 1/m*temp1;
        else
        temp2 = temp2 + (h(i)-y(i))*X(i,j);
        grad(j)= 1/m * temp2 +(lambda / m *theta(j));
        end
    end
        
end
end





% =============================================================

