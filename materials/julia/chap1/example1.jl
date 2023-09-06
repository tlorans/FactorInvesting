scoringfunction(x, max, min) = 2 * (x - min) / (max - min) - 1;


C = [34.2; 
    65.4; 
    12.3; 
    32.7; 
    98.1; 
    7.4;
    2.6;
    120.9;
    12.4;
    56.0];


max_x = maximum(C);
min_x = minimum(C);

S = map(x -> scoringfunction(x, max_x, min_x), C);