clear;

% logical vector for portfolio belonging
long_portfolio = logical([
    0;
    0;
    0;
    0;
    1;
    0;
    0;
    1;
    0
    ]);

% Count the number of 'true' values
num_true = sum(long_portfolio);

% Calculate the equal weight for each 'true' value
equal_weight = 1 / num_true;

% Create the weight vector
weight_vec = zeros(1, length(long_portfolio));
weight_vec(long_portfolio) = equal_weight;