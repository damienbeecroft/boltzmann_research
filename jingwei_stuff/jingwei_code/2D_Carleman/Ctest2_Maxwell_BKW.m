%clc
clear all
% test 2D Maxwell molecule

gamma = 0;
b_gamma = 1/(2*pi);


N = 64;

S = 3;
R = 2*S;

L = (3*sqrt(2)+1)/2*S;
Ntheta = 4;

dv = 2*L/N;
v = (-L+dv/2):dv:(L-dv/2);
[vv1,vv2] = ndgrid(v);


t = 0.5;
K = 1-exp(-t/8)/2;
dK = exp(-t/8)/16;

f = 1/(2*pi*K^2)*exp(-(vv1.^2+vv2.^2)/(2*K)).*(2*K-1+(1-K)/(2*K)*(vv1.^2+vv2.^2));
df = (-2/K+(vv1.^2+vv2.^2)/(2*K^2)).*f...
   + 1/(2*pi*K^2)*exp(-(vv1.^2+vv2.^2)/(2*K)).*(2-1/(2*K^2)*(vv1.^2+vv2.^2));
df = df*dK;
extQ = df;
Max = 1/(2*pi)*exp(-(vv1.^2+vv2.^2)/2);

figure(1); set(gca,'FontSize',14);
plot(v,f(:,N/2),'k','LineWidth',1.5); title('f'); hold on
plot(v,Max(:,N/2),'r','LineWidth',1.5);

figure(2); set(gca,'FontSize',14);
plot(v,extQ(:,N/2),'k','LineWidth',1.5); title('Q(f)'); hold on


Q = CBoltz2_Carl_Maxwell(f,N,R,L,Ntheta)*b_gamma;
absmax = max(max(abs(extQ-Q)))
figure(2); plot(v,Q(:,N/2),'o');