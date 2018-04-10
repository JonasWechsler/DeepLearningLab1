function P = EvaluateClassifier(X, W, b)
    WX = W*X;
    S = bsxfun(@plus, WX, b);
    P = soft_max(S')';
    P(isnan(P)) = 0;
end

function mu = soft_max(eta)
    % mu(i,c) = exp(eta(i,c))/sum_c' exp(eta(i,c'))
    % This function is from matlabtools.googlecode.com
    c = 3;

    tmp = exp(c*eta);
    denom = sum(tmp, 2);
    mu = bsxfun(@rdivide, tmp, denom);
end