X1 = randn (50 ,2) + 1;
X2 = randn (51 ,2) - 1;
Y1 = ones (50 ,1);
Y2 = -ones (51 ,1);
disp (X1);
figure ;
hold on;
plot (X1 (: ,1) , X1 (: ,2) , 'ro');
plot (X2 (: ,1) , X2 (: ,2) , 'bo');
X = [X1;X2];
Y=[Y1;Y2];
MdlLinear = fitcdiscr(X,Y);
K = MdlLinear.Coeffs(1,2).Const;  
L = MdlLinear.Coeffs(1,2).Linear;
f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h2 = fimplicit(f);
h2.Color = 'g';
h2.LineWidth = 1;
h2.DisplayName = 'Boundary Line';


