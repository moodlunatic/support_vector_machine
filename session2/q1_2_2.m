X =(-3:0.01:3)';%原来是0.2
Y=sinc(X)+0.1.*randn(length(X),1);
Xtrain = X (1:2: end);
Ytrain = Y (1:2: end);
Xtest = X (2:2: end);
Ytest = Y (2:2: end);

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






