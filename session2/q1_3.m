
X = 6.* rand (100 , 3) - 3;
Y = sinc (X(: ,1)) + 0.1.* randn (100 ,1) ;
[gam,sig2] = tunelssvm({X,Y,'f',[],[],'RBF_kernel'},'simplex','leaveoneoutlssvm',{'mse'});
[selected , ranking] = bay_lssvmARD ({X, Y,'f', gam , sig2});
