function [VpSat,VsSat,RHOBSat]=applico_RPM_new(dati_r_m)
% data_r_m : Matrix of Vp,Vs and Rho respectively
Kw=2.5e9;
rhow=1045;
Kg=1e9; %0.2
rhog=900;

VpSat=zeros(size(dati_r_m,1),1);
VsSat=zeros(size(dati_r_m,1),1);
RHOBSat=zeros(size(dati_r_m,1),1);

for i=1:size(dati_r_m,1) 
    if dati_r_m(i,3)>.50
        PhiC=0.4;
        clay=dati_r_m(i,3);
        phi=dati_r_m(i,2);
        sw=dati_r_m(i,1);        
    else
        PhiC=0.4;
        clay=dati_r_m(i,3);
        phi=dati_r_m(i,2);
        sw=dati_r_m(i,1);        
    end
    
    invKf=(1-sw)/Kg + (sw/Kw);
    Kf=1/invKf;
    rhof=rhog*(1-sw)+rhow*(sw);    
    [~,~,~,~,VpSat(i),VsSat(i),RHOBSat(i)]=softsediments(clay,0,0,80,PhiC,12,Kf,rhof,0.8,phi);
                                     %Clay,Feldspar,Calcite,Pressure,PhiC,Coord,Kf,RHOf,Sfact,Phi
                                     % Change Coord from 12 to .....10
end
VpSat=VpSat;
VsSat=VsSat;
RHOBSat=RHOBSat;

end