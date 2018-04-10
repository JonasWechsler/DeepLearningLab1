function Assignment1
    main()
end 

function [X, Y, y] = loadBatch(filename)
    N = 10000;
    d = 3072;
    K = 10;
    
    inf = load(filename);
    X = double(inf.data.');
    y = inf.labels;
    assert(isequal(size(X), [d N]));
    assert(isequal(size(y), [N 1]));
    Y = bsxfun(@eq, y(:), 0:max(y)).';
    assert(isequal(size(Y), [K N]));
end

function [W, b] = init_model(K, d)
    std_dev = 0.01;
    W = std_dev*randn(K, d);
    b = std_dev*randn(K, 1);
    assert(isequal(size(W), [K d]));
    assert(isequal(size(b), [K 1]));
end

function k = Predict(X, W, b)
    [~, k] = max(EvaluateClassifier(X, W, b));
    k = k';
end

function acc = ComputeAccuracy(X, y, W, b)
    P = Predict(X, W, b);
    acc = sum(bsxfun(@eq, P, y))/length(P);
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    g = -(Y - P).';
    coef = 1/size(X,2);
    grad_W = coef*(g.'*X.') + 2*lambda*W;
    grad_b = coef*sum(g.',2);
end

function [batch_X, batch_Y] = Sample(X, Y, batch_size)
    idx = randperm(size(X, 2), batch_size);
    batch_X = X(:,idx);
    batch_Y = Y(:,idx);
end

function v = max_diff(A, B)
    D = A - B;
    v = max(abs(D(:)));
end

function main
    [X, Y, y] = loadBatch("data_batch_1.mat");
    [K, ~] = size(Y);
    [d, N] = size(X);
    [W, b] = init_model(K, d);
    P = EvaluateClassifier(X, W, b);
    assert(isequal(size(P), [K N]));
    [grad_W, grad_b] = ComputeGradients(X, Y, P, W, 0.01);
    assert(isequal(size(grad_W), size(W)));
    assert(isequal(size(grad_b), size(b)));
    for iter = 1:10
        [batch_X, batch_Y] = Sample(X, Y, 1);
        P = EvaluateClassifier(batch_X, W, b);
        [grad_W, grad_b] = ComputeGradients(batch_X, batch_Y, P, W, 0);
        [sgrad_b, sgrad_W] = ComputeGradsNumSlow(batch_X, batch_Y, W, b, 0, 1e-6);
        [ngrad_b, ngrad_W] = ComputeGradsNum(batch_X, batch_Y, W, b, 0, 1e-6);
        %fprintf("%i %i %i %i %i %i\n", max(sgrad_W(:)), max(ngrad_W(:)), max(grad_W(:)), max(sgrad_b(:)), max(ngrad_b(:)), max(grad_b(:)))
        fprintf("%i %i %i %i\n", max_diff(sgrad_b, ngrad_b), max_diff(sgrad_W, ngrad_W), max_diff(grad_W, sgrad_W), max_diff(grad_b, sgrad_b))
    end
end