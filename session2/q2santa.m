load santafe.mat;
lag = 50;%也就是order
Xu = windowize (Z, 1:lag + 1);
Xtra = Xu(1:end-lag,1:lag);
Ytra = Xu(1:end-lag,end);
Xs=Z(end-lag+1:end,1); %starting point for iterative prediction
[gam,sig2] = tunelssvm({Xtra,Ytra,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'mae'});
[alpha ,b] = trainlssvm ({Xtra,Ytra, 'f', gam , sig2,'RBF_kernel' });

nb = 200;
prediction = predict ({Xtra,Ytra, 'f', gam , sig2,'RBF_kernel' }, Xs , nb);
figure ;
hold on;
plot (Ztest , 'k');
plot (prediction , 'r');
hold off;
mse(Ztest-prediction)



 
 
 