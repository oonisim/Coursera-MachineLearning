function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


%------------------------------------------------------------------------
% Calculate the logistic value of the hidden layer H / Activation 2 (A2). 
%------------------------------------------------------------------------
X = [ones(m, 1) X]; % bias

H_NET = X * Theta1';
H = sigmoid(H_NET);
H = [ones(size(H, 1), 1) H]; % bias

%------------------------------------------------------------------------
% Calculate the logistic value of the output layer O / Activation 3 (A3).
%------------------------------------------------------------------------
O_NET = H * Theta2';
O = sigmoid(O_NET);

%------------------------------------------------------------------------
% Convert y (10, 10, 10 .... 9, 9, 9, ... ,1) into a boolean matrix.
% if y(i) is i, then E(i, :) is [1,0,0,0,0,0,0,0,0,0,0].
%------------------------------------------------------------------------
Y = zeros(m, num_labels);
E = eye(num_labels);
for i = 1:m
    Y(i, :) = E(y(i), :);
end

%------------------------------------------------------------------------
% Calculate the cost at output without regularization.
%------------------------------------------------------------------------
J = 0;
one = ones(m, num_labels);

% Each row of (Y .* log(O)) is the cost at each output node for input xi. 
% Take the sum of all columns in a row by sum(v, 2) for the cost of each xi.
cost_y1 = -1 * sum(Y .* log(O), 2) / m;
cost_y0 = -1 * sum((one - Y) .* log(one - O), 2) / m;

% Each row of (cost_y1 + cost_y0) is the cost of xi. 
% Take the sum of all rows for the total cost (xi: i = 1,2,3..)
J = sum(cost_y1 + cost_y0);

t1_square=Theta1(:, 2:end).^2;
t2_square=Theta2(:, 2:end).^2;
reg_theta1= sum(t1_square(:)) * lambda / (2 * m);
reg_theta2= sum(t2_square(:)) * lambda / (2 * m);
J = J + ( reg_theta1 + reg_theta2 )

% ====================== YOUR CODE HERE ======================
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

for i = 1:m  
    % i is training set index of X (including bias). X(i, :) is 401 data.
    xi = X(i, :);
    yi = Y(i, :);
    
    % hi is the i th output of the hidden layer. H(i, :) is 26 data.
    hi = H(i, :);
    
    % oi is the i th output layer. O(i, :) is 10 data.
    oi = O(i, :);
    
    %------------------------------------------------------------------------
    % Calculate the gradients of Theta2
    %------------------------------------------------------------------------
    delta_theta2 = oi - yi;
    Theta2_grad = Theta2_grad + bsxfun(@times, hi, transpose(delta_theta2));
 
    %------------------------------------------------------------------------
    % Calculate the gradients of Theta1
    %------------------------------------------------------------------------
    % Derivative of g(z): g'(z)=g(z)(1-g(z)) where g(z) is sigmoid(H_NET).
    dgz = (hi .* (1 - hi));
    delta_theta1 = dgz .* sum(bsxfun(@times, Theta2, transpose(delta_theta2)));
    % There is no input into H0, hence there is no theta for H0. Remove H0.
    delta_theta1 = delta_theta1(2:end);
    Theta1_grad = Theta1_grad + bsxfun(@times, xi, transpose(delta_theta1));

    %------------------------------------------------------------------------
    % Iterative version to calculate the gradients of Theta1
    %------------------------------------------------------------------------
    %alpha=0; % Input layer index alpha (including bias) for Theat1_grad(j, alpha)
    %for alpha = 1:size(xi, 2)
    %    j = 1; % Hidden layer index j (including bias) 
    %    for j = 2:size(hi,2)
    %        t = xi(alpha) * ( hi(j) * (1 - hi(j) ) );
    %        e = 0;
    %        for k = 1:size(oi, 2)
    %            e = e + Theta2(k, j) * (oi(k) - yi(k));
    %        end
    %        g1 = t * e;
    %        Theta1_grad(j-1, alpha) = Theta1_grad(j-1, alpha) + g1;
    %    end
    %end    
    
end

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta2_grad_reg= [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)] .* (lambda / m);
Theta2_grad = (Theta2_grad / m) + Theta2_grad_reg;

Theta1_grad_reg= [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)] .* (lambda / m);
Theta1_grad = (Theta1_grad / m) + Theta1_grad_reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
