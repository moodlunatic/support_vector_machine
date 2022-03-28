load ripley;
%数据可视化
figure;
gscatter(Xtrain(:,1),Xtrain(:,2),Ytrain);
title('Scatter of Ripley Data');
type='classification';
L_fold = 10; % L-fold crossvalidation

%Linear Kernel
gam = tunelssvm({Xtrain,Ytrain,type,[],[],'lin_kernel'},'simplex','crossvalidatelssvm',{L_fold,'misclass'});
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'});
plotlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'},{alpha,b});
[Yest,Y_latent] = simlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'},{alpha,b},Xtest);
roc(Y_latent,Ytest);


%POlynomina Kernel
[gam,sig2] = tunelssvm({Xtrain,Ytrain,type,[],[],'poly_kernel'},'simplex','crossvalidatelssvm',{L_fold,'misclass'});
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'poly_kernel'});
plotlssvm({Xtrain,Ytrain,type,gam,sig2,'poly_kernel'},{alpha,b});
[Yest,Y_latent] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'poly_kernel'},{alpha,b},Xtest);
roc(Y_latent,Ytest);

%RBF Kernel
[gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,type,[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{L_fold,'misclass'});
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'},{alpha,b});
[Yest,Y_latent] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'},{alpha,b},Xtest);
roc(Y_latent,Ytest);