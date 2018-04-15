function P = EvaluateClassifier(X, W, b)
    WX = W*X;
    S = bsxfun(@plus, WX, b);
    P = soft_max(S);
end

function mu = soft_max(eta)
    tmp = exp(eta);
    tmp(isinf(tmp)) = 1e100;
    denom = sum(tmp, 1);
    mu = bsxfun(@rdivide, tmp, denom);
end