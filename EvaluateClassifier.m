function [P, H, S1] = EvaluateClassifier(X, W, b)
    [W1, W2] = W{:};
    [b1, b2] = b{:};
    S1 = bsxfun(@plus, W1*X, b1);
    H = max(0, S1);
    S = bsxfun(@plus, W2*H, b2);
    P = soft_max(S);
end

function mu = soft_max(eta)
    tmp = exp(eta);
    tmp(isinf(tmp)) = 1e100;
    denom = sum(tmp, 1);
    mu = bsxfun(@rdivide, tmp, denom);
end