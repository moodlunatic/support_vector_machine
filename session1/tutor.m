X = 2.*rand(100,2)-1;
Y = sign(sin(X(:,1))+X(:,2));
gam = 10;
sig2 = 0.4;
type = 'classification';
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});
Xt = 2.*rand(10,2)-1;
Ytest = simlssvm({X,Y,type,gam,sig2,'RBF_kernel'},{alpha,b},Xt);
plotlssvm({X,Y,type,gam,sig2,'RBF_kernel'},{alpha,b});