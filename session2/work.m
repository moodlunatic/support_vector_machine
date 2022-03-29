
X=(-3:0.01:3)';
Y = sinc(X') + 0.1.* randn(length(X), 1);

Xtrain = X(1:2:length(X));
Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X));
Ytest = Y(2:2:length(Y));

%%
sigs = [0.01 1 100]; gammas=[10 10^3 10^6];
figure;
for i=1:length(gammas)
    gam = gammas(i);
    sig2 = sigs(i);
    mdl_in = {Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'};
    [alpha,b] = trainlssvm(mdl_in);
 
    subplot(4, 2, i);
    plotlssvm(mdl_in, {alpha,b});
  
    YtestEst = simlssvm(mdl_in, {alpha,b},Xtest);
    plot(Xtest,Ytest,'b.');
    hold on;
    plot(Xtest,YtestEst,'g+');
    %legend('Ytest','YtestEst');
    title(['sig2=' num2str(sig2) ',gam=' num2str(gam)]);
    hold off;
end


