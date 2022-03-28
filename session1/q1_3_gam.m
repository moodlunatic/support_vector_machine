type='c'; 
disp('RBF kernel')
gamlist = [0.01 0.1 1 5 10 50 100]; 
sig2=1;

errlist=[];

for gam=gamlist,
    disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
    err = sum(Yht~=Ytest); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)
    disp('Press any key to continue...'), pause,         
end


%
% make a plot of the misclassification rate wrt. sig2
%
figure;
plot(log(gamlist), errlist, '*-'), 
xlabel('log(gam)'), ylabel('number of misclass'),

figure;
plot(log(gamlist), errlist/ length(Ytest), '*-'), 
xlabel('log(gam)'), ylabel('error rate'),


