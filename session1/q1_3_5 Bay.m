load iris
[gam ,sig2 , cost] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'},'gridsearch', 'crossvalidatelssvm',{10, 'misclass'});
[alpha , b]=trainlssvm ({ Xtrain , Ytrain , 'c', gam , sig2,'RBF_kernel'});
[Yest , Ylatent ] = simlssvm ({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'}, {alpha, b}, Xtest);
roc(Ylatent,Ytest);
bay_modoutClass ({ Xtrain , Ytrain , 'c', gam , sig2 }, 'figure');
bay_modoutClass ({ Xtrain , Ytrain , 'c', 0.06 , 0.0433 }, 'figure');