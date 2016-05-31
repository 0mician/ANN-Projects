clc, clear all, close all;
addpath 'export_fig'; % export pdf: https://github.com/altmany/export_fig
rng(7); % setting random seed

% student number = r0575791, datasets initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
csv_import = importdata('../datasets/winequality-white.csv');
data = csv_import.data;
cpos = data(data(:,12) == 6,:); cpos_size = length(cpos);
cneg = data(data(:,12) == 7,:); cneg_size = length(cneg);

X = [cpos(:,1:(end-1)) ; cneg(:,1:(end-1))]'; % label removed (6,7)
Y = [ones(cpos_size,1) ; zeros(cneg_size, 1)]'; % replaced by 1 and 0
stdx = mapstd(X);

% creation of training, validation, and test sets
n = cpos_size + cneg_size; 
[train_ind, val_ind, test_ind] = dividerand(n, 0.8, 0.1, 0.1);

% Neural net training
%%%%%%%%%%%%%%%%%%%%%
i = 1; ccr_val = zeros(10,1); ccr_test = zeros(10,1);
for hnn = [1 2 3 4 5 10 15 20 25 30]
    net = patternnet(hnn);%
    net.divideFcn = 'divideind';
    net.trainParam.showWindow = false;
    net.divideParam.trainInd = train_ind;
    net.divideParam.valInd = val_ind;
    net.divideParam.testInd = test_ind;
    net = train(net, stdx, Y);

    pred = round(sim(net, stdx(:,val_ind)));
    ccr_val(i) = sum(pred == Y(:,val_ind))*100/length(val_ind);
    pred = round(sim(net, stdx(:,test_ind)));
    ccr_test(i) = sum(pred == Y(:,test_ind))*100/length(test_ind);
    i = i + 1;
end

% PCA
%%%%%
[coeff,scores,~,~,explained] = pca(X(:,train_ind)','Centered', true, 'VariableWeights','variance');
figure('Color',[1 1 1]);
subplot(1,2,1);
scatter3(scores(:,1), scores(:,2), scores(:,3), 15, Y(train_ind));
title('Graph of the first 3 PC'); xlabel('PC 1'); ylabel('PC 2'); zlabel('PC3');
subplot(1,2,2);
bar(explained);
title('Principal components importance'); xlabel('PC number'); ylabel('Variability explained (%)');

% reconstruction of whole dataset with 6 components
reduced_dim = coeff(:,1:6);
reduced_data = X' * reduced_dim;
reduced_data = reduced_data';
stdxr = reduced_data;%mapstd(reduced_data);

i = 1; ccr_val = zeros(10,1); ccr_test = zeros(10,1);
for hnn = [1 2 3 4 5 10 15 20 25 30]
    net = patternnet(hnn);%
    net.divideFcn = 'divideind';
    net.trainParam.max_fail = 50;
    net.trainParam.min_grad=1e-10;
    net.trainParam.showWindow = true;
    net.divideParam.trainInd = train_ind;
    net.divideParam.valInd = val_ind;
    net.divideParam.testInd = test_ind;
    net = train(net, stdxr, Y);

    pred = round(sim(net, stdxr(:,val_ind)));
    ccr_val(i) = sum(pred == Y(:,val_ind))*100/length(val_ind);
    pred = round(sim(net, stdxr(:,test_ind)));
    ccr_test(i) = sum(pred == Y(:,test_ind))*100/length(test_ind);
    i = i + 1;
end

[max_ccr_test, pos_test] = max(ccr_test);
[max_ccr_val, pos_val] = max(ccr_val);

