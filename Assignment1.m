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
    acc = double(sum(bsxfun(@eq, P, y)))/double(length(P));
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    g = - (Y - P).';
    coef = 1/size(X,2);
    grad_W = coef*(g.'*X.') + 2*lambda*W;
    grad_b = coef*sum(g.',2);
end

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
end

function DisplayTemplates(W, b)
    for i=1:10
        im = reshape(W(i, :), 32, 32, 3);
        s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
        s_im{i} = permute(s_im{i}, [2, 1, 3]);
    end
    montage(s_im, 'Size', [1 10]);
end

function [W, b] = train(X, Y, y, W, b, n_batch, eta, n_epochs, lambda, n_test)
    [K, ~] = size(Y);
    [d, N] = size(X);
    train_X = X(:,1:N-n_test);
    train_Y = Y(:,1:N-n_test);
    train_y = y(1:N-n_test);
    test_X = X(:,N-n_test+1:N);
    test_Y = Y(:,N-n_test+1:N);
    test_y = y(N-n_test+1:N);
    plot_x = zeros(n_epochs);
    plot_train_accuracy = zeros(n_epochs);
    plot_train_cost = zeros(n_epochs);
    plot_test_accuracy = zeros(n_epochs);
    plot_test_cost = zeros(n_epochs);
    
    best_W = W;
    best_b = b;
    best_accuracy = 0;
    
    for iter = 1:n_epochs
        [W, b] = epoch(train_X, train_Y, train_y, W, b, n_batch, eta, lambda);
        
        if 0
            eta = eta*0.999;
        end
        
        train_accuracy = ComputeAccuracy(train_X, train_y, W, b);
        train_cost = ComputeCost(train_X, train_Y, W, b, lambda);
        test_accuracy = ComputeAccuracy(test_X, test_y, W, b);
        test_cost = ComputeCost(test_X, test_Y, W, b, lambda);
        
        plot_train_accuracy(iter) = train_accuracy;
        plot_train_cost(iter) = train_cost;
        plot_test_accuracy(iter) = test_accuracy;
        plot_test_cost(iter) = test_cost;
        
        if test_accuracy > best_accuracy
           best_accuracy = test_accuracy;
           best_W = W;
           best_b = b;
        end
        
        plot_x(iter) = iter;
    end
    if 0
        W = best_W;
        b = best_b;
    end
    fprintf("%.4f (end) vs %.4f (best)\n",ComputeAccuracy(test_X, test_y, W, b), best_accuracy);
    %{
    plot(plot_x, plot_train_accuracy)
    title(sprintf('n batch: %i, eta: %i, n epochs: %i, lambda: %i', n_batch, eta, n_epochs, lambda))
    xlabel('Epoch Number')
    ylabel('Training Accuracy')
    figure()
    plot(plot_x, plot_test_accuracy)
    title(sprintf('n batch: %i, eta: %i, n epochs: %i, lambda: %i', n_batch, eta, n_epochs, lambda))
    xlabel('Epoch Number')
    ylabel('Test Accuracy')
    figure()
    plot(plot_x, plot_train_cost)
    title(sprintf('n batch: %i, eta: %i, n epochs: %i, lambda: %i', n_batch, eta, n_epochs, lambda))
    xlabel('Epoch Number')
    ylabel('Training Cost')
    figure()
    plot(plot_x, plot_test_cost)
    title(sprintf('n batch: %i, eta: %i, n epochs: %i, lambda: %i', n_batch, eta, n_epochs, lambda))
    xlabel('Epoch Number')
    ylabel('Test Cost')
    %}
end

function learner_main
    [X, Y, y] = loadBatch("data_batch_1.mat");
    
    if 0
        for i = 2:5
           filename = sprintf("data_batch_%d.mat", i);
            [X0, Y0, y0] = loadBatch(filename);
            X = [X X0];
            Y = [Y Y0];
            y = [y; y0];
        end
    end
    
    [K, ~] = size(Y);
    [d, N] = size(X);
    [W, b] = init_model(K, d);
    [W, b] = train(X, Y, y, W, b, 100, 0.1, 40, 0, 1000);
    [W, b] = train(X, Y, y, W, b, 100, 0.01, 40, 0, 1000);
    [W, b] = train(X, Y, y, W, b, 100, 0.01, 40, 0.1, 1000);
    [W, b] = train(X, Y, y, W, b, 100, 0.01, 40, 1, 1000);
    DisplayTemplates(W, b)
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