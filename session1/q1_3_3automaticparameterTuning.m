load iris
%[gam ,sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'},'simplex', 'crossvalidatelssvm',{10, 'misclass'});

[gam2 ,sig22 , cost2 ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'},'gridsearch', 'crossvalidatelssvm',{10, 'misclass'});

