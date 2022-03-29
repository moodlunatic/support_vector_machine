X =(-3:0.2:3)';
Y=sinc(X)+0.1.*randn(length(X),1);
Xtrain = X (1:2: end);
Ytrain = Y (1:2: end);
Xtest = X (2:2: end);
Ytest = Y (2:2: end);

gam =1000000;
sig2 = 100;
type = 'function estimation';

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
Yt = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);
figure; 
plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
hold on;
scatter(Xtest,Ytest,'r')
mse_result=mse(Ytest-Yt)

%调参
mselist=[];
gamlist=[];
sig2list=[];

for i=1:10
    type = 'function estimation';
    alg='gridsearch';
    [gam,sig2] = tunelssvm({Xtrain,Ytrain,type,[],[],'RBF_kernel'},alg,'leaveoneoutlssvm',{'mse'});
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
    Yt = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);
    figure; 
    plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
    hold on;
    scatter(Xtest,Ytest,'r');
    mse_result=mse(Ytest-Yt);
    mselist=[mselist;mse_result];
    gamlist=[gamlist;gam];
    sig2list=[sig2list;sig2];
end
%再现结果
gam =25.1761;
sig2 = 0.2068;
type = 'function estimation';

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
Yt = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);
figure; 
plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
hold on;
scatter(Xtest,Ytest,'r')
mse_result=mse(Ytest-Yt)



