function Q = CBoltz2_Carl_Maxwell(f,N,R,L,Ntheta)
% Carleman spectral method for the classical Boltzmann collision operator
% 2D Maxwell molecule
% N # of Fourier modes: f(N,N), Q(N,N)
% theta: mid-point rule

tic
[l1,l2] = ndgrid([0:N/2-1,-N/2:-1]);
FTf = fft2(f);

QG = zeros(N,N);
bb = zeros(N,N);

wtheta = pi/Ntheta;
theta = wtheta/2:wtheta:(pi-wtheta/2);
sig1 = cos(theta);
sig2 = sin(theta);

for q = 1:Ntheta
    aa1 = alpha2(l1*sig1(q)+l2*sig2(q),R,L);
    aa2 = alpha2(sqrt(l1.^2+l2.^2-(l1*sig1(q)+l2*sig2(q)).^2),R,L);
    
    QG = QG + 2*wtheta*ifft2(aa1.*FTf).*ifft2(aa2.*FTf);
    bb = bb + 2*wtheta*aa1.*aa2;
end

QL = f.*ifft2(bb.*FTf);

Q = real(QG-QL);

fprintf(1, 'time of CBoltz2_Carl_Maxwell is %4.2f sec\n', toc);