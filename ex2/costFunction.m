function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

z=theta'*X';
h = 1./(1+(exp(-z)));
temp0 = 0;
temp1=0;


for i=1:m
    temp0 = temp0 + [(-1)*y(i)*log(h(i))-(1-y(i))*log(1-h(i))];
    temp1 = temp1 + (h(i)-y(i))*X(i,1:3);

% =============================================================

end
J = 1/m * temp0;
grad = 1/m*temp1;

