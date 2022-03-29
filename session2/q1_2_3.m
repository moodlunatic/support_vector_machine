X = 6.* rand (100 , 3) - 3;
Y = sinc (X(: ,1)) + 0.1.* randn (100 ,1) ;
Xtrain = X;
Ytrain = Y;

sig2 = 0.4;
gam = 10;
crit_L1 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 1);
crit_L2 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 2);
crit_L3 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 3);
%The model can be optimized with respect to these criteria:
[~, alpha ,b] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 1);
[~, gam] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 2);
[~, sig2 ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 3);
%For regression, the error bars can be computed using Bayesian inference using
sig2e = bay_errorbar ({ Xtrain , Ytrain , 'f', gam , sig2 }, 'figure');

[selected , ranking ]= bay_lssvmARD ({X, Y, 'f', gam , sig2 });


%下面是自己尝试去做cross
costTotal= crossvalidate({Xtrain(:,1),Ytrain, 'f',gam,sig2})
cost1 = crossvalidate({Xtrain(:,1),Ytrain, 'f',gam,sig2})
cost2 = crossvalidate({Xtrain(:,2),Ytrain, 'f',gam,sig2})
cost3 = crossvalidate({Xtrain(:,3),Ytrain, 'f',gam,sig2})
plot([1,2,3,4],costall)
set(gca,'XTickLabel',{'Three input','Input1','Input2','Input3'})
xlabel('Input')
ylabel('cost')






