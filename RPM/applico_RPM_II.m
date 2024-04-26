function [VpSat,VsSat,RHOBSat]=applico_RPM_II(dati_r_m)

Kw=2.5e9;
rhow=1045;
Kg=1e8; %0.2
rhog=600;

VpSat=zeros(size(dati_r_m,1),1);
VsSat=zeros(size(dati_r_m,1),1);
RHOBSat=zeros(size(dati_r_m,1),1);

for i=1:size(dati_r_m,1) 
    if dati_r_m(i,3)>.50
        PhiC=0.7;
        clay=dati_r_m(i,3)+eps;
        phi=dati_r_m(i,2)+eps;
        sw=dati_r_m(i,1)+eps;        
    else
        PhiC=0.45;
        clay=dati_r_m(i,3)+eps;
        phi=dati_r_m(i,2)+eps;
        sw=dati_r_m(i,1)+eps;        
    end
    
    invKf=(1-sw)/Kg + (sw/Kw);
    Kf=1/invKf;
    rhof=rhog*(1-sw)+rhow*(sw);    
    [~,~,~,~,VpSat(i),VsSat(i),RHOBSat(i)]=softsediments(clay,0,0,20,PhiC,6,Kf,rhof,0.5,phi);
    
end
VpSat=VpSat;
VsSat=VsSat;
RHOBSat=RHOBSat;

end