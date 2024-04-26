function d_pre=calcola_dati_pre_stack(Vp,Vs,Rho,wavelet,ang)

[M,N,Q]=size(Vp);
d_pre=single(zeros(M,N,length(ang),Q)); % single precison variable 

for i=1:Q
    for j=1:N
        pippo=CMP_zoepprtiz([Vp(:,j,i) Vs(:,j,i) Rho(:,j,i)],wavelet,ang);
        pippo=[pippo(1,:);pippo];        
        d_pre(:,j,1:length(ang),i)=single(pippo);
    end    
end

end