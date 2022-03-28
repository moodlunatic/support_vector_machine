load iris
%
% Train the LS-SVM classifier using polynomial kernel
%
type='c'; 
gam = 1; 
t = 1; 
degreelist=[1, 2, 3, 4, 5];
errorlist=[0,0,0,0,0];
for i=1:5,
degree = degreelist(i);
fprintf('Polynomial kernel of degree %d\n',i),

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'});

figure; plotlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});

[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xtest);

err = sum(Yht~=Ytest); 
errorlist(i)=err/length(Ytest);
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)
end
plot(degreelist,errorlist);
title("Error rate of polynomial kernel")
xlabel("Degree");
ylabel("Error rate");

      
    

