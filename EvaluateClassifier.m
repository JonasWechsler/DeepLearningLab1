function P = EvaluateClassifier(X, W, b)
    WX = W*X;
    S = bsxfun(@plus, WX, b);
    P = soft_max(S);
end

function mu = soft_max(eta)
    tmp = exp(eta);
    denom = sum(tmp, 1);
    mu = bsxfun(@rdivide, tmp, denom);
end