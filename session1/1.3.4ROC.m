load iris;
%数据可视化
figure;
gscatter(Xtrain(:,1),Xtrain(:,2),Ytrain);
gscatter(Xtest(:,1),Xtest(:,2),Ytest);
title('Scatter of Ripley Data');
type='classification';
L_fold = 10; % L-fold crossvalidation

%RBF Kernel
[gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,type,[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{L_fold,'misclass'});
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'},{alpha,b});
[Yest,Y_latent] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'},{alpha,b},Xtest);
roc(Y_latent,Ytest);
Y_latent = latentlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'},{alpha,b},Xtest);
roc(Y_latent,Ytest);