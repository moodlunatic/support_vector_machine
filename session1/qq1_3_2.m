load iris

% train LS-SVM classifier with linear kernel 

type='c'; 

% use RBF kernel
%
perflist=[];
gamlist = [0.1,1,10,100,1000] ;
sig2list =[0.001,0.01,0.1,1,10,100,1000];
%crosss
for sig2=sig2list,
    for gam=gamlist,
    perf = rsplitvalidate({ Xtrain , Ytrain , 'c', gam , sig2, 'RBF_kernel'}, 0.80, 'misclass');
    perflist=[perflist;perf];
    end
end

x=log(sig2huatu');
y=log(gamhuatu');

z=perflist;
[X,Y,Z]=griddata(x,y,z,linspace(min(x),max(x),25)',linspace(min(y),max(y),25),'v4');
surf(X,Y,Z);
xlabel('log(sig2)');
ylabel('log(gam)');
zlabel('performance');

