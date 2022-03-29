X = linspace(-1,1,50)';
Y = (15*(X.^2-1).^2.*X.^4).*exp(-X)+normrnd(0,0.1,length(X),1);
type = 'function estimation';
