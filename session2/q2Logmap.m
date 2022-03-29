load logmap.mat;
orders=[5:5:100];
maelist=[];
sig2list=[];
gamlist=[];
for order=orders
    X = windowize (Z, 1:( order + 1));
    Y = X(:, end);
    X = X(:, 1: order );
    [gam,sig2] = tunelssvm({X,Y,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'mae'});
    [alpha ,b] = trainlssvm ({X, Y, 'f', gam , sig2 });
    Xs = Z(end - order +1: end , 1);
    nb = 50;
    prediction = predict ({X, Y, 'f', gam , sig2 'RBF_kernel'}, Xs , nb);
    figure ;
    hold on;
    plot (Ztest , 'k');
    plot (prediction , 'r');
    hold off;
    mae1=mae(Ztest-prediction);
    maelist=[maelist;mae1];
    sig2list=[sig2list;sig2];
    gamlist=[gamlist;gam];
end

%结果再现
order=25; %
X = windowize (Z, 1:( order + 1));
Y = X(:, end);
X = X(:, 1: order );
[gam,sig2] = tunelssvm({X,Y,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'mae'});
[alpha ,b] = trainlssvm ({X, Y, 'f', gam , sig2 });
Xs = Z(end - order +1: end , 1);
nb = 50;
prediction = predict ({X, Y, 'f', gam , sig2 'RBF_kernel'}, Xs , nb);
figure ;
hold on;
plot (Ztest , 'k');
plot (prediction , 'r');
hold off;

 
 
 