clear
a0 = 0.000531632;
a1 = 0.0207337;
a2 = 0.254652;
b0 = 0.00122807;
alpha1 = -0.3;
alpha2 = -0.55;
alpha3 = 0.4;
alpha4 = 0.21;

% solving for y(u)
syms x1 x2 x3 u y
eq1 = -a2*x1 + x2 == 0;
eq2 = -a1*x1 + x3 == 0;
eq3 = -a0*x1+b0*(alpha1*u+alpha2*u^2+alpha3*u^3+alpha4*u^4) == 0;
eq4 = y == x1;
vars = [x1,x2,x3,y];
eqns = [eq1, eq2, eq3, eq4];
sol = solve(eqns, vars);
yu = sol.y;

% plotting y(u)
figure(1)
fplot(yu, [-1,1])
hold on

% linearizing y(u) i u0 point
u0 = 0; % linearization point
dy_du = diff(yu);
yu_u0 = subs(yu, u0);
dyu_u0 = subs(dy_du, u0);
lin_yu = yu_u0 + dyu_u0*(u-u0);

% ploting y(u0)
fplot(lin_yu, [-1,1])
xlabel('u') 
ylabel('y') 
%print("y(u)u0=0.5.png","-dpng","-r400")

% dyskretny
T = 5;
