clear;

rankFunction = @(x, sortedScores) find(x == sortedScores, 1, 'first');

S = [-0.4658; 
    0.0617; 
    -0.8360; 
    -0.4911; 
    0.6145; 
    -0.9189;
    -1;
    1;
    -0.8343;
    -0.0972];

sorted_scores = sort(S, 'descend');

rank_vector = arrayfun(@(x) rankFunction(x, sorted_scores), S);


