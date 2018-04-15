function Assignment1
    learner_main()
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

function [W, b] = init_model(K, d)
    std_dev = 0.01;
    W = std_dev*randn(K, d);
    b = std_dev*randn(K, 1);
    assert(isequal(size(W), [K d]));
    assert(isequal(size(b), [K 1]));
end

function k = Predict(X, W, b)
    P = EvaluateClassifier(X, W, b);
    [~, k] = max(P);
    k = k' - 1;
end

function acc = ComputeAccuracy(X, y, W, b)
    P = Predict(X, W, b);
    acc = double(sum(bsxfun(@eq, P, y)))/length(P);
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    g = - (Y - P).';
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

function [W, b] = epoch(X, Y, y, W, b, n_batch, eta, lambda)
    N = size(X, 2);
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        batch_X = X(:, inds);
        batch_Y = Y(:, inds);
        P = EvaluateClassifier(batch_X, W, b);
        [grad_W, grad_b] = ComputeGradients(batch_X, batch_Y, P, W, lambda);
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
        [batch_X, batch_Y] = Sample(X, Y, 3);
        P = EvaluateClassifier(batch_X, W, b);
        [grad_W, grad_b] = ComputeGradients(batch_X, batch_Y, P, W, 0);
        [sgrad_b, sgrad_W] = ComputeGradsNumSlow(batch_X, batch_Y, W, b, 0, 1e-6);
        [ngrad_b, ngrad_W] = ComputeGradsNum(batch_X, batch_Y, W, b, 0, 1e-6);
        %disp(grad_W)
        %fprintf("%i %i %i %i %i %i\n", max(sgrad_W(:)), max(ngrad_W(:)), max(grad_W(:)), max(sgrad_b(:)), max(ngrad_b(:)), max(grad_b(:)))
        fprintf("%i %i\n", max_diff(grad_W, sgrad_W), max_diff(grad_b, sgrad_b))
        W = W - grad_W*0.01;
        b = b - grad_b*0.01;
    end
end