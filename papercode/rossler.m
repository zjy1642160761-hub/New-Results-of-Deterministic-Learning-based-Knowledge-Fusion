function xdot=rossler(t,x)
global p1 p2 p3;
% p1=0.15 ;p2=0.15;p3=1;

x1=x(1);
x2=x(2);
x3=x(3);

dx1=-x2-x3;
dx2=x1+p1*x2;
dx3=p2+x3*(x1-p3);

xdot=[dx1;dx2;dx3];
t