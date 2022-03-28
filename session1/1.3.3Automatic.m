load iris;
%数据可视化
type='classification';
L_fold = 10; % L-fold crossvalidation
errlist=[];
%RBF Kernel simplex
[gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,type,[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{L_fold,'misclass'});
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'},{alpha,b});
 % Obtain the output of the trained classifier
[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
err = sum(Yht~=Ytest); errlist=[errlist; err];
fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100);

%RBF Kernel gridsearch
[gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,type,[],[],'RBF_kernel'},'gridsearch','crossvalidatelssvm',{L_fold,'misclass'});
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'},{alpha,b});
 % Obtain the output of the trained classifier
[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
err = sum(Yht~=Ytest); errlist=[errlist; err];
fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100);

