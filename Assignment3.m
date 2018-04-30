function Assignment3
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

function [W, b] = init_model(layers)
    std_dev = 0.001;
    W = {};
    b = {};
    for i = 1:length(layers)-1
        W{i} = std_dev*randn(layers(i+1), layers(i));
        b{i} = std_dev*randn(layers(i+1), 1);
    end
end


function [phi] = activation(X)
    phi = max(0, X);
    %{
    a = exp(-X);
    b = a + 1;
    c = 2/b;
    phi = c - 1;
    %}
end

function [delta_phi] = delta_activation(X)
    delta_phi = (X > 0);
    %delta_phi = (1 + activation(X)).*(1 - activation(X))/2;
end

function g = BatchNormBackPass(g, S, mu, var)
    e =  2.2204e-16;
    V = diag(var + e)^(-1/2);
    n = size(g, 1);
    
    grad_v = zeros(size(var.'));
    for idx = 1:n
        grad_v = grad_v -0.5 * g(idx,:) * V^3 * diag(S(:,idx)-mu);
    end
    
    grad_m = 0;
    for idx = 1:n
        grad_m = grad_m - g(idx,:)*V;
    end
    
    for idx = 1:size(g,1)
        g(idx,:) = g(idx,:)*V + (2/n)*grad_v*diag(S(:,idx)-mu) + (1/n)*grad_m;
    end
end

function [grad_W, grad_b, mu, var] = ComputeGradients(X_in, Y, W, b, lambda)
    k = length(W);
    grad_b = cell(k);
    grad_W = cell(k);
    [P, X, S, S_hat, mu, var] = EvaluateClassifier(X_in, W, b);
    coef = 1/size(X_in,2);
    g = - (Y - P).';
    for i = k:-1:2
        grad_W{i} = coef*(g.'*X{i}.') + 2*lambda*W{i};
        grad_b{i} = coef*sum(g.',2);
        g = g*W{i};
        g = g.*delta_activation(S_hat{i-1}.');
        g = BatchNormBackPass(g, S{i-1}, mu{i-1}, var{i-1});
    end
    grad_W{1} = coef*(g.'*X{1}.') + 2*lambda*W{1};
    grad_b{1} = coef*sum(g.',2);
end

function [batch_X, batch_Y] = Sample(X, Y, batch_size)
    idx = randperm(size(X, 2), batch_size);
    batch_X = X(:,idx);
    batch_Y = Y(:,idx);
end

function k = Predict(X, W, b, mu, var)
    [P, ~] = EvaluateClassifier(X, W, b, mu, var);
    [~, k] = max(P);
    k = k' - 1;
end

function acc = ComputeAccuracy(X, y, W, b, mu, var)
    P = Predict(X, W, b, mu, var);
    acc = double(sum(bsxfun(@eq, P, y)))/length(P);
end

function J = ComputeCost(X, Y, W, b, lambda, mu, var)
    L = CrossEntropyLoss(X, Y, W, b, mu, var);
    sum_squared = 0;
    for idx=1:length(W)
        sum_squared = sum_squared + sum(W{idx}(:).^2);
    end
    J = mean(L) + lambda*sum_squared;
end

function L = CrossEntropyLoss(X, Y, W, b, mu, var)
    [P, ~] = EvaluateClassifier(X, W, b, mu, var);
    YTP = sum(Y'.*P',2);
    L = -1*arrayfun(@log, YTP);
end

function S = BatchNormalize(S, avg, var)
    e =  2.2204e-16;
    S = (diag(var + e)^(-1/2))*(S - avg);
end

function [P, X, S, S_hat, mu, var_S] = EvaluateClassifier(X_in, W, b, mu, var_S)
    k = length(W);
    X = cell(k+1);
    S = cell(k);
    S_hat = cell(k);
    
    calc_avg = ~exist('mu', 'var');
    assert(calc_avg == ~exist('var_S', 'var'));
    
    if calc_avg
        mu = cell(k-1);
        var_S = cell(k-1);
    end
    
    N = size(X_in, 2);
    X{1} = X_in;
    
    for l = 1:k-1
        S{l} = bsxfun(@plus, W{l}*X{l}, b{l});
        
        if calc_avg
            mu{l} = (1/N)*sum(S{l},2);
            var_S{l} = var(S{l}, 0, 2);
            var_S{l} = var_S{l} * (N-1)/N;
        end
        
        S_hat{l} = BatchNormalize(S{l}, mu{l}, var_S{l});
        
        X{l+1} = activation(S_hat{l});
    end
    S{k} = bsxfun(@plus, W{k}*X{k}, b{k});
    P = soft_max(S{k});
end

function mu = soft_max(eta)
    tmp = exp(eta);
    tmp(isinf(tmp)) = 1e100;
    denom = sum(tmp, 1);
    mu = bsxfun(@rdivide, tmp, denom);
end

function [W, b, momentum_W, momentum_b, momentum_mu, momentum_v] = epoch(X, Y, y, W, b, n_batch, eta, lambda, rho, momentum_W, momentum_b, momentum_mu, momentum_v)
    N = size(X, 2);
    
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        batch_X = X(:, inds);
        batch_Y = Y(:, inds);
        [grad_W, grad_b, mu, var] = ComputeGradients(batch_X, batch_Y, W, b, lambda);
        
        if isequal(size(momentum_mu), [0,0])
               momentum_mu = mu;
               momentum_v = var;
        end
        
        for m = 1:length(mu)
           momentum_mu{m} = 0.99*momentum_mu{m} + (1-0.99)*mu{m};
           momentum_v{m} = 0.99*momentum_v{m} + (1-0.99)*var{m};
        end
        
        for m = 1:length(W)
            momentum_W{m} = rho*momentum_W{m} + eta*grad_W{m};
            momentum_b{m} = rho*momentum_b{m} + eta*grad_b{m};
        end
        
        for m = 1:length(W)
            W{m} = W{m} - momentum_W{m};
            b{m} = b{m} - momentum_b{m};
        end
    end
    

end

function [W, b] = train(X, Y, y, n_batch, eta, n_epochs, lambda, n_nodes, rho, eta_decay, test_X, test_Y, test_y, eta_decay_rate)
    fprintf("batch size: %i, eta: %.4f, epochs: %i, lambda: %i, nodes: %i, rho: %.2f, eta_decay: %.2f, eta_rate: %i\n", n_batch, eta, n_epochs, lambda, n_nodes, rho, eta_decay, eta_decay_rate);
    [K, ~] = size(Y);
    [d, ~] = size(X);
    
    [W, b] = init_model([d n_nodes K]);
        
    best_W = W;
    best_b = b;
    best_accuracy = 0;
    
    momentum_W = cell(length(W));
    momentum_b = cell(length(b));
    momentum_mu = {};
    momentum_v = {};
    
    for i=1:length(W)
       momentum_W{i} = zeros(size(W{i}));
       momentum_b{i} = zeros(size(b{i}));
    end
    
    for iter = 1:n_epochs
        [W, b, momentum_W, momentum_b, momentum_mu, momentum_v] = epoch(X, Y, y, W, b, n_batch, eta, lambda, rho, momentum_W, momentum_b, momentum_mu, momentum_v);
        if mod(iter, eta_decay_rate) == 0
            eta = eta*eta_decay;
        end
        
        cost = ComputeCost(test_X, test_Y, W, b, lambda, momentum_mu, momentum_v);
        accuracy = ComputeAccuracy(test_X, test_y, W, b, momentum_mu, momentum_v);
        tr_accuracy = ComputeAccuracy(X, y, W, b, momentum_mu, momentum_v);
        tr_cost = ComputeCost(X, Y, W, b, lambda, momentum_mu, momentum_v);
        fprintf("%i,%i,%i,%i\n", accuracy, cost, tr_accuracy, tr_cost);
        if cost > 3*2.3
           return 
        end
        
        if accuracy > best_accuracy
            best_accuracy = accuracy;
            best_W = W;
            best_b = b;
        end
    end
    
    W = best_W;
    b = best_b;
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
    
    [W, b] = train(train_X, train_Y, train_y, 200, 0.0449, 30, 4.419090e-09, 50, 0.9, 0.1, test_X, test_Y, test_y, 10);
    accuracy = ComputeAccuracy(test_X, test_y, W, b)*100;
    cost = ComputeCost(test_X, test_Y, W, b, 0);
    tr_accuracy = ComputeAccuracy(train_X, train_y, W, b)*100;
    tr_cost = ComputeCost(train_X, train_Y, W, b, 0);
    fprintf("%i, %i, %i, %i\n", accuracy, cost, tr_accuracy, tr_cost);
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
    
    min_log_eta = log(0.025)/log(10);
    max_log_eta = log(0.04)/log(10);
    min_log_lambda = -6;
    max_log_lambda = -15;
    
    fileID = fopen('2.csv','w');
    
    best_cost_eta = -1;
    best_cost_lambda = -1;
    best_cost = 10;
    best_acc_eta = -1;
    best_acc_lambda = -1;
    best_acc = 0;
    
    batch_size = 200;
    n_epochs = 30;
    hidden_layer = 200;
    rho = 0.9;
    eta_decay = 0.99;
    
    fprintf(fileID, "batch_size: %i, epochs: %i, hidden: %i, rho: %i, decay: %i\n", batch_size, n_epochs, hidden_layer, rho, eta_decay);
    fprintf(fileID, "eta, lambda, accuracy, cost\n");
    for t = 1:50
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

function v = max_diff(A, B)
    v = 0;
    for idx = 1:length(A)
        a = A{idx};
        b = B{idx};
        d = a - b;
        v_idx = max(abs(d(:)));
        v = max(v, v_idx);
    end
end

function tester_main
    [X, Y, ~] = loadBatch("data_batch_1.mat");
    X = X(1:100,:);
    Y = Y(1:10,:);
    [X, ~] = zero_mean(X);
    [K, ~] = size(Y);
    [d, ~] = size(X);
    n_nodes = 50;
    [W, b] = init_model([d n_nodes K]);
    for iter = 1:10
        [batch_X, batch_Y] = Sample(X, Y, 3);
        [grad_W, grad_b] = ComputeGradients(batch_X, batch_Y, W, b, 0);
        [sgrad_b, sgrad_W] = ComputeGradsNumSlow(batch_X, batch_Y, W, b, 0, 1e-5);
        [ngrad_b, ngrad_W] = ComputeGradsNum(batch_X, batch_Y, W, b, 0, 1e-5);
        fprintf("%i,%i,%i,%i\n", max_diff(grad_W, sgrad_W), max_diff(grad_b, sgrad_b), max_diff(grad_W, ngrad_W), max_diff(grad_b, ngrad_b));
    end
end

function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

c = ComputeCost(X, Y, W, b, lambda);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b{j}(i) = (c2-c) / h;
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})   
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
        
        grad_W{j}(i) = (c2-c) / h;
    end
end

end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
end
