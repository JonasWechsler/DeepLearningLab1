function J = ComputeCost(X, Y, W, b, lambda)
    L = CrossEntropyLoss(X, Y, W, b);
    W2 = W.^2;
    J = mean(L) + lambda*sum(W2(:));
end

function L = CrossEntropyLoss(X, Y, W, b)
    P = EvaluateClassifier(X, W, b);
    YTP = sum(Y'.*P',2);
    L = -1*arrayfun(@log, YTP);
end

