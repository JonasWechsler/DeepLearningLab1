function Assignment1
    simple_learner_main();
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

function [W, b, momentum] = epoch(X, Y, y, W, b, n_batch, eta, lambda, rho, momentum)
    [W1, W2] = W{:};
    [b1, b2] = b{:};
    N = size(X, 2);
    
    parameters = {zeros(size(W1)), zeros(size(W2)), zeros(size(b1)), zeros(size(b2))};
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        batch_X = X(:, inds);
        batch_Y = Y(:, inds);
        [grad_W, grad_b] = ComputeGradients(batch_X, batch_Y, W, b, lambda);
        [grad_W1, grad_W2] = grad_W{:};
        [grad_b1, grad_b2] = grad_b{:};
        grads = {grad_W1, grad_W2, grad_b1, grad_b2};
        for m = 1:4
            momentum{m} = rho*momentum{m} + eta*grads{m};
        end

        W1 = W1 - momentum{1};
        W2 = W2 - momentum{2};
        b1 = b1 - momentum{3};
        b2 = b2 - momentum{4};

        W = {W1, W2};
        b = {b1, b2};
    end
    

end

function [W, b] = train(X, Y, y, n_batch, eta, n_epochs, lambda, n_nodes, rho, eta_decay, test_X, test_Y, test_y)
    fprintf("batch size: %i, eta: %.4f, epochs: %i, lambda: %i, nodes: %i, rho: %.2f, eta_decay: %.2f\n", n_batch, eta, n_epochs, lambda, n_nodes, rho, eta_decay);
    [K, ~] = size(Y);
    [d, ~] = size(X);
    
    [W, b] = init_model(K, n_nodes, d);
    [W1, W2] = W{:};
    [b1, b2] = b{:};
        
    best_W = W;
    best_b = b;
    best_accuracy = 0;
    
    momentum = {zeros(size(W1)), zeros(size(W2)), zeros(size(b1)), zeros(size(b2))};
    
    for iter = 1:n_epochs
        [W, b, momentum] = epoch(X, Y, y, W, b, n_batch, eta, lambda, rho, momentum);
        eta = eta*eta_decay;
        
        
        cost = ComputeCost(test_X, test_Y, W, b, lambda);
        accuracy = ComputeAccuracy(test_X, test_y, W, b);
        fprintf("%i,%i\n", accuracy, cost);
        if cost > 3*2.3
           return 
        end
    end
end

function simple_learner_main
    [X, Y, y] = loadBatch("data_batch_1.mat");
    if 1
        for i = 2:5
            filename = sprintf("data_batch_%d.mat", i);
            [X0, Y0, y0] = loadBatch(filename);
            X = [X X0];
            Y = [Y Y0];
            y = [y; y0];
        end
    end
    
    
    train_size = size(X,2)-1000;
    
    train_X = X(:,1:train_size);
    [train_X, mean_X] = zero_mean(train_X);
    train_Y = Y(:,1:train_size);
    train_y = y(1:train_size);
    
    test_X = X(:,train_size:size(X,2));
    test_X = test_X - mean_X;
    test_Y = Y(:,train_size:size(Y,2));
    test_y = y(train_size:size(y));
    
    [W, b] = train(train_X, train_Y, train_y, 200, 0.04487559, 30, 0.00000000441909, 50, 0.9, 0.95, test_X, test_Y, test_y);
    accuracy = ComputeAccuracy(test_X, test_y, W, b)*100;
    cost = ComputeCost(test_X, test_Y, W, b, 0);
    fprintf("%i, %i\n", accuracy, cost);
end

function learner_main
    [X, Y, y] = loadBatch("data_batch_1.mat");
    train_size = size(X,2)-1000;
    
    train_X = X(:,1:train_size);
    [train_X, mean_X] = zero_mean(train_X);
    train_Y = Y(:,1:train_size);
    train_y = y(1:train_size);
    
    test_X = X(:,train_size:size(X,2));
    test_X = test_X - mean_X;
    test_Y = Y(:,train_size:size(Y,2));
    test_y = y(train_size:size(y));
    
    eta = 0.01;
    lambda = 0.00001;
    
    min_log_eta = log(0.04)/log(10);
    max_log_eta = log(0.06)/log(10);
    min_log_lambda = -15;
    max_log_lambda = -8;
    
    fileID = fopen('1.csv','w');
    
    best_cost_eta = -1;
    best_cost_lambda = -1;
    best_cost = 10;
    best_acc_eta = -1;
    best_acc_lambda = -1;
    best_acc = 0;
    
    batch_size = 200;
    n_epochs = 10;
    hidden_layer = 50;
    rho = 0.9;
    eta_decay = 0.99;
    
    fprintf(fileID, "batch_size: %i, epochs: %i, hidden: %i, rho: %i, decay: %i\n", batch_size, n_epochs, hidden_layer, rho, eta_decay);
    fprintf(fileID, "eta, lambda, accuracy, cost\n");
    for t = 1:100
        e = min_log_eta + (max_log_eta - min_log_eta)*rand(1,1);
        eta = 10^e;
        e = min_log_lambda + (max_log_lambda - min_log_lambda)*rand(1,1);
        lambda = 10^e;
        
        [W, b] = train(train_X, train_Y, train_y, batch_size, eta, n_epochs, lambda, hidden_layer, rho, eta_decay, test_X, test_Y, test_y);
        accuracy = ComputeAccuracy(test_X, test_y, W, b)*100;
        cost = ComputeCost(test_X, test_Y, W, b, 0);
        fprintf(fileID, "%i, %i, %i, %i\n", eta, lambda, accuracy, cost);
        
        if accuracy > best_acc
           best_acc = accuracy;
           best_acc_eta = eta;
           best_acc_lambda = lambda;
        end
        
        if cost < best_cost
           best_cost = cost;
           best_cost_eta = eta;
           best_cost_lambda = lambda;
        end
    end
    
    fprintf("(Cost: %i) best eta: %i, best lambda: %i\n", best_cost, best_cost_eta, best_cost_lambda);
    fprintf("(Accuracy: %i) best eta: %i, best lambda: %i\n", best_acc, best_acc_eta, best_acc_lambda);
    
    %{
    fprintf("Train Accuracy: %.2f%%\n", 100*ComputeAccuracy(train_X, train_y, W, b));
    fprintf("Train Cost:     %i\n", ComputeCost(train_X, train_Y, W, b, 0));
    fprintf("Test Accuracy:  %.2f%%\n", 100*ComputeAccuracy(test_X, test_y, W, b));
    fprintf("Test Cost:      %i\n", ComputeCost(test_X, test_Y, W, b, 0));
    %}
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
    P = EvaluateClassifier(X, W, b);
    assert(isequal(size(P), [K N]));
    [grad_W, grad_b] = ComputeGradients(X, Y, W, b, 0);
    [grad_W1, grad_W2] = grad_W{:};
    [grad_b1, grad_b2] = grad_b{:};
    [W1, W2] = W{:};
    [b1, b2] = b{:};
    assert(isequal(size(grad_W1), size(W1)));
    assert(isequal(size(grad_W2), size(W2)));
    assert(isequal(size(grad_b1), size(b1)));
    assert(isequal(size(grad_b2), size(b2)));
    for iter = 1:10
        [batch_X, batch_Y] = Sample(X, Y, 10);
        [grad_W, grad_b] = ComputeGradients(batch_X, batch_Y, W, b, 0);
        [sgrad_b, sgrad_W] = ComputeGradsNumSlow(batch_X, batch_Y, W, b, 0, 1e-5);
        [ngrad_b, ngrad_W] = ComputeGradsNum(batch_X, batch_Y, W, b, 0, 1e-5);
        fprintf("%i,%i,%i,%i\n", max_diff(grad_W, sgrad_W), max_diff(grad_b, sgrad_b), max_diff(grad_W, ngrad_W), max_diff(grad_b, ngrad_b));
        
        %fprintf("%i,%i\n", max_diff(grad_W, ngrad_W), max_diff(grad_b, ngrad_b));
        
        [grad_W1, grad_W2] = grad_W{:};
        [grad_b1, grad_b2] = grad_b{:};
        W1 = W1 - 0.01*grad_W1;
        W2 = W2 - 0.01*grad_W2;
        b1 = b1 - 0.01*grad_b1;
        b2 = b2 - 0.01*grad_b2;
        
        %W = {W1 W2};
        %b = {b1 b2};
    end
end