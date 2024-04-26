function data=CMP_zoepprtiz(m,ric,ang)

% m - a matrix of vp,vs,rho of size(2x3) ideally
N=size(m,1)-1;
Rpp=zeros(N,length(ang));
data=Rpp;

for j=1:N
    Rpp(j,:)=real(avopp(m(j,1),m(j,2),m(j,3),m(j+1,1),...
        m(j+1,2),m(j+1,3),ang,1));
end

%{
for j=1:length(ang)
    
    n=nextpow2(size(Rpp,1));
    p=2^n;
    RPP=fft(Rpp(:,j),p);
    RIC=fft(ric,p);
    keyboard
    DATA=abs(RPP).*abs(RIC).*exp(sqrt(-1)*angle(RIC));
    keyboard
    data(:,j)=real(ifft(DATA,size(Rpp,1)));
end
%}


for j=1:length(ang)
    data(:,j)=conv(Rpp(:,j),ric(:,j),'same');
end


return