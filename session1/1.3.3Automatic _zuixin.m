load iris;
%数据可视化
type='classification';
L_fold = 10; % L-fold crossvalidation
gamlist=[];
sig2list=[];
costlist=[];
%RBF Kernel simplex
tic;
for i=1:8
    [gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,type,[],[],'RBF_kernel'},'gridsearch','crossvalidatelssvm',{L_fold,'misclass'});
    gamlist=[gamlist;gam];
    sig2list=[sig2list;sig2];
    costlist=[costlist;cost];
end
toc;




