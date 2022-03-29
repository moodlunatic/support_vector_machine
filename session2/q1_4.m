X = ( -6:0.2:6)';
Y = sinc (X) + 0.1.* rand ( size (X));
%Outliers can be added via:
out = [15 17 19];
Y( out) = 0.7+0.3* rand (size( out));
out = [41 44 46];
Y( out) = 1.5+0.2* rand ( size ( out));
%Let’s say we first train a LS-SVM regressor model, without giving special attention to the
%outliers. We can implement this as:
model = initlssvm (X, Y, 'f', [], [], 'RBF_kernel');
costFun = 'crossvalidatelssvm';
model = tunelssvm (model, 'simplex', costFun , {10 , 'mse';});
plotlssvm (model);

%If we were to train a robust LS-SVM model, using robust crossvalidation, we can implement this as:
model = initlssvm (X, Y, 'f', [], [], 'RBF_kernel');
costFun = 'rcrossvalidatelssvm';
wFun = 'whuber';
model = tunelssvm (model , 'simplex', costFun , {10 ,'mae';}, wFun );
model = robustlssvm ( model );
plotlssvm ( model );
subtitle(wFun);
%用mae


figure;
wFun = 'whampel';
model = tunelssvm (model , 'simplex', costFun , {10 ,'mae';}, wFun );
model = robustlssvm ( model );
plotlssvm ( model );
subtitle(wFun);

figure;
wFun = 'wlogistic';
model = tunelssvm (model , 'simplex', costFun , {10 ,'mae';}, wFun );
model = robustlssvm ( model );
plotlssvm ( model );
subtitle(wFun);

figure;
wFun = 'wmyriad';
model = tunelssvm (model , 'simplex', costFun , {10 ,'mae';}, wFun );
model = robustlssvm ( model );
plotlssvm ( model );
subtitle(wFun);








