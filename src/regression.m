clc, clear all, close all;
addpath 'export_fig'; % export pdf: https://github.com/altmany/export_fig
load '../datasets/regression.mat';

% generation of own data from student number r0575791
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Tnew = (9*T1 + 7*T2 + 7*T3 + 5*T4 + 5*T5)/(9 + 7 + 7 + 5 + 5);

% plotting the dataset
%%%%%%%%%%%%%%%%%%%%%%
figure('Color', [1 1 1]);
tri = delaunay(X1, X2);
h = trisurf(tri, X1, X2, Tnew);
l = light('Position', [-50 -15 29]);
lighting phong; shading interp; 
colormap Jet; colorbar EastOutside;
xlabel('X1','FontSize',14); 
ylabel('X2','FontSize',14);
zlabel('f(X1,X2)','FontSize',14);

export_fig('regression_dataset.pdf');

% random permutation of the data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataset = [X1 X2 Tnew];
dataset = dataset(randperm(size(dataset,1)),:);
ratio = 1/3;
[trainInd, valInd, testInd] = dividerand(3000, ratio, ratio, ratio);

% training set definition
Xtrain = [dataset(trainInd,1)'; dataset(trainInd,2)'];
Ytrain = dataset(trainInd,3)';

% validation set definition
Xval = [dataset(valInd,1)'; dataset(valInd,2)'];
Yval = dataset(valInd,3)';

% test set definition
Xtest = [dataset(testInd,1)'; dataset(testInd,2)'];
Ytest = dataset(testInd,3)';

% plotting surfaces
figure('Color', [1 1 1]);

% plot train set
subplot(1,3,1);
tri = delaunay(Xtrain(:,1), Xtrain(:,2));
h = trisurf(tri, Xtrain(:,1), Xtrain(:,2), Ytrain);
l = light('Position', [-50 -15 29]);
lighting phong; colormap Jet;
xlabel('Xtrain1','FontSize',14); 
ylabel('Xtrain2','FontSize',14);
zlabel('f(Xtrain1,Xtrain2)','FontSize',14);

% plot validation set
subplot(1,3,2);
tri = delaunay(Xval(:,1), Xval(:,2));
h = trisurf(tri, Xval(:,1), Xval(:,2), Yval);
l = light('Position', [-50 -15 29]);
lighting phong; colormap Jet;
xlabel('Xval1','FontSize',14); 
ylabel('Xval2','FontSize',14);
zlabel('f(Xval1,Xval2)','FontSize',14);

% plot test set
subplot(1,3,3);
tri = delaunay(Xtest(:,1), Xtest(:,2));
h = trisurf(tri, Xtest(:,1), Xtest(:,2), Ytest);
l = light('Position', [-50 -15 29]);
lighting phong; colormap Jet; colorbar EastOutside;
xlabel('Xtest1','FontSize',14); 
ylabel('Xtest2','FontSize',14);
zlabel('f(Xtest1,Xtest2)','FontSize',14);

export_fig('regression_trainingsets.pdf');

% Estimation of number of hidden neurons
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% with sigmoid transfer function
n_hiddenn = 2:2:200;
mse_validation_logsig = zeros(100,1);
i = 1;
for nn = n_hiddenn
    net = feedforwardnet(nn);
    net.trainParam.showWindow = false;
    net.trainParam.epochs = 5000;
    net.divideFcn = 'dividetrain';
    net.layers{1}.transferFcn = 'logsig';
    net = train(net, Xtrain, Ytrain, 'UseParallel', 'yes');
    pred = sim(net, Xval);
    mse_validation_logsig(i) = perform(net, Yval, pred);
    i = i + 1;
end

% with hyperbolic tangent transfer function
n_hiddenn = 2:2:200;
mse_validation_tansig = zeros(100,1);
i = 1;
for nn = n_hiddenn
    net = feedforwardnet(nn);
    net.trainParam.showWindow = false;
    net.trainParam.epochs = 5000;
    net.divideFcn = 'dividetrain';
    net.layers{1}.transferFcn = 'tansig';
    net = train(net, Xtrain, Ytrain, 'UseParallel', 'yes');
    pred = sim(net, Xval);
    mse_validation_tansig(i) = perform(net, Yval, pred);
    i = i + 1;
end

figure('Color', [1 1 1]);
subplot(1,2,1);
semilogy(5:5:100, mse_validation_tansig);
hold on;
semilogy(5:5:100, mse_validation_logsig);
grid on;
xlabel('Hidden neurons','FontSize',14); 
ylabel('Validation MSE','FontSize',14);

% Evaluation on test set
%%%%%%%%%%%%%%%%%%%%%%%%

net = feedforwardnet(60);
net.trainParam.showWindow = true;
net.trainParam.epochs = 5000;
net.divideFcn = 'dividetrain';
net.layers{1}.transferFcn = 'tansig';
net = train(net, Xtrain, Ytrain, 'UseParallel', 'yes');
pred = sim(net, Xtest);
performance = perform(net, Ytest, pred);

hold on;
plot(60, performance, 'r*');
legend('tansig', 'logsig', 'Test MSE (60 neurons)');

export_fig('regression_logtan_error.pdf');

% Plot param selection + Plot surface of test set + prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tets MSE for parameter selection
figure('Color', [1 1 1]);
subplot(1,2,1);
semilogy(5:5:100, mse_validation_tansig);
hold on;
semilogy(5:5:100, mse_validation_logsig);
grid on;
xlabel('Hidden neurons','FontSize',14); 
ylabel('Validation MSE','FontSize',14);
hold on;
plot(60, performance, 'r*');
legend('tansig', 'logsig', 'Test MSE (60 neurons)');

% surface plot + prediction
subplot(1,2,2);
tri = delaunay(Xtest(1,:), Xtest(2,:));
h = trisurf(tri, Xtest(1,:), Xtest(2,:), Ytest);
l = light('Position', [-50 -15 29]);
lighting phong; colormap winter; colorbar EastOutside;
xlabel('Xtest1','FontSize',14); 
ylabel('Xtest2','FontSize',14);
zlabel('f(Xtest1,Xtest2)','FontSize',14);
hold on;
scatter3(Xtest(1,:), Xtest(2,:), pred, 'r', 'filled');