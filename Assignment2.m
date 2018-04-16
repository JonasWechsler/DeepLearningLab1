function Assignment1
    tester_main()
end 

function [X, Y, y] = loadBatch(filename)
    N = 10000;
    d = 3072;
    K = 10;
    
    inf = load(filename);
    X = double(inf.data.')/256;
    y = inf.labels;
    assert(isequal(size(X), [d N]));
    assert(isequal(size(y), [N 1]));
    Y = bsxfun(@eq, y(:), 0:max(y)).';
    assert(isequal(size(Y), [K N]));
end

function [X, mean_X] = zero_mean(X)
    mean_X = mean(X, 2);
    X = X - repmat(mean_X, [1, size(X, 2)]);
end

function [W, b] = init_model(K, m, d)
    std_dev = 0.001;
    W1 = std_dev*randn(m, d);
    b1 = std_dev*randn(m, 1);
    W2 = std_dev*randn(K, m);
    b2 = std_dev*randn(K, 1);
    W = {W1 W2};
    b = {b1 b2};
    assert(isequal(size(W1), [m d]));
    assert(isequal(size(b1), [m 1]));
    assert(isequal(size(W2), [K m]));
    assert(isequal(size(b2), [K 1]));
end

function k = Predict(X, W, b)
    [P, ~, ~] = EvaluateClassifier(X, W, b);
    [~, k] = max(P);
    k = k' - 1;
end

function acc = ComputeAccuracy(X, y, W, b)
    P = Predict(X, W, b);
    acc = double(sum(bsxfun(@eq, P, y)))/length(P);
end

function [grad_W, grad_b] = ComputeGradients(X, Y, W, b, lambda)
    [W1, W2] = W{:};
    [b1, b2] = b{:};
    [P, H, S1] = EvaluateClassifier(X, W, b);
    g = - (Y - P).';
    coef = 1/size(X,2);
    grad_W2 = coef*(g.'*H.') + 2*lambda*W2;
    grad_b2 = coef*sum(g.',2);
    g = g*W2;
    g = g.*(S1.' > 0);%for each column in g, ...
    grad_W1 = coef*(g.'*X.') + 2*lambda*W1;
    grad_b1 = coef*sum(g.',2);
    grad_W = {grad_W1 grad_W2};
    grad_b = {grad_b1 grad_b2};
    assert(isequal(size(W1), size(grad_W1)));
    assert(isequal(size(W2), size(grad_W2)));
    assert(isequal(size(b1), size(grad_b1)));
    assert(isequal(size(b2), size(grad_b2)));
end

%{
function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    g = - (Y - P).';
    coef = 1/size(X,2);
    grad_W = coef*(g.'*X.') + 2*lambda*W;
    grad_b = coef*sum(g.',2);
end
%}

function [batch_X, batch_Y] = Sample(X, Y, batch_size)
    idx = randperm(size(X, 2), batch_size);
    batch_X = X(:,idx);
    batch_Y = Y(:,idx);
end

function v = max_diff(WA, WB)
    [W1A, W2A] = WA{:};
    [W1B, W2B] = WB{:};
    D1 = W1A - W1B;
    D2 = W2A - W2B;
    v1 = max(abs(D1(:)));
    v2 = max(abs(D2(:)));
    v = max(v1, v2);
end

function [W, b] = epoch(X, Y, y, W, b, n_batch, eta, lambda)
    N = size(X, 2);
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        batch_X = X(:, inds);
        batch_Y = Y(:, inds);
        [grad_W, grad_b] = ComputeGradients(batch_X, batch_Y, W, b, lambda);
        W = W - eta*grad_W;
        b = b - eta*grad_b;
    end
    fprintf("%i %i\n", ComputeAccuracy(X, y, W, b), ComputeCost(X, Y, W, b, lambda));
end

function [W, b] = train(X, Y, y, n_batch, eta, n_epochs, lambda)
    [K, ~] = size(Y);
    [d, ~] = size(X);
    [W, b] = init_model(K, d);
    for iter = 1:n_epochs
        [W, b] = epoch(X, Y, y, W, b, n_batch, eta, lambda);
    end
end

function learner_main
    [X, Y, y] = loadBatch("data_batch_1.mat");
    [W, b] = train(X, Y, y, 100, 0.01, 40, 0);
end

function tester_main
    [X, Y, ~] = loadBatch("data_batch_1.mat");
    X = X(1:100,:);
    Y = Y(1:10,:);
    [X, X_mean] = zero_mean(X);
    [K, ~] = size(Y);
    [d, N] = size(X);
    n_nodes = 50;
    [W, b] = init_model(K, n_nodes, d);
    [W1, W2] = W{:};
    [b1, b2] = b{:};
    P = EvaluateClassifier(X, W, b);
    assert(isequal(size(P), [K N]));
    [grad_W, grad_b] = ComputeGradients(X, Y, W, b, 0);
    [grad_W1, grad_W2] = grad_W{:};
    [grad_b1, grad_b2] = grad_b{:};
    assert(isequal(size(grad_W1), size(W1)));
    assert(isequal(size(grad_W2), size(W2)));
    assert(isequal(size(grad_b1), size(b1)));
    assert(isequal(size(grad_b2), size(b2)));
    for iter = 1:10
        [batch_X, batch_Y] = Sample(X, Y, 3);
        [grad_W, grad_b] = ComputeGradients(batch_X, batch_Y, W, b, 0);
        [sgrad_b, sgrad_W] = ComputeGradsNumSlow(batch_X, batch_Y, W, b, 0, 1e-5);
        fprintf("%i %i\n", max_diff(grad_W, sgrad_W), max_diff(grad_b, sgrad_b))
        [ngrad_b, ngrad_W] = ComputeGradsNum(batch_X, batch_Y, W, b, 0, 1e-5);
        fprintf("%i %i\n", max_diff(grad_W, ngrad_W), max_diff(grad_b, ngrad_b))
    end
end