# Define the rank function
function rankFunction(x, sortedScores)
    return findfirst(==(x), sortedScores)
end

# Given vector S
S = [-0.4658, 
     0.0617, 
    -0.8360, 
    -0.4911, 
     0.6145, 
    -0.9189,
    -1.0,
     1.0,
    -0.8343,
    -0.0972]

# Sort the scores in descending order
sorted_scores = sort(S, rev=true)

# Calculate the rank vector
rank_vector = map(x -> rankFunction(x, sorted_scores), S)