clear
a0 = 0.000531632;
a1 = 0.0207337;
a2 = 0.254652;
b0 = 0.00122807;
alpha1 = -0.3;
alpha2 = -0.55;
alpha3 = 0.4;
alpha4 = 0.21;

u0 = 0.5;
fun = alpha1 + 2*alpha2*u0 + 3*alpha3*u0^2 + 4*alpha4*u0^3;
A = [-a2 1 0; -a1 0 1; -a0 0 0];
B = [0; 0; b0*fun];
C = [1 0 0];
D = 0;
[NUM, DEN] = ss2tf(A, B, C, D);
G = tf(NUM, DEN);
static_gain = dcgain(G);