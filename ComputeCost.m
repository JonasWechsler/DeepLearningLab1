function J = ComputeCost(X, Y, W, b, lambda)
    L = CrossEntropyLoss(X, Y, W, b);
    [W1, W2] = W{:};
    sum_squared = sum(W1(:).^2) + sum(W2(:).^2);
    J = mean(L) + lambda*sum_squared;
end

function L = CrossEntropyLoss(X, Y, W, b)
    [P, ~, ~] = EvaluateClassifier(X, W, b);
    YTP = sum(Y'.*P',2);
    L = -1*arrayfun(@log, YTP);
end