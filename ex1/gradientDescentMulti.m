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

    d0=0;
    d1=0;
    d2=0;
    for i=1:m,
            d0=d0 + theta(1,1)+theta(2,1)*X(i,2) + theta(3,1)*X(i,3) - y(i) ;
            d1=d1 + ( theta(1,1)+theta(2,1)*X(i,2) + theta(3,1)*X(i,3) - y(i) ) * X(i,2) ;
	    d2=d2 + ( theta(1,1)+theta(2,1)*X(i,3) + theta(3,1)*X(i,3) - y(i) ) * X(i,3); 

    end;
    temp1 = theta(1,1) - (alpha*d0)/m;
    temp2 = theta(2,1) - (alpha*d1)/m;
    temp3 = theta(3,1) - (alpha*d2)/m;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

    theta(1,1) = temp1;
    theta(2,1) = temp2;
    theta(3,1) = temp3;

end

end
