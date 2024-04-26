function [Phi,VpDry,VsDry,RHOBDry,VpSat,VsSat,RHOBSat]=softsediments(Clay,Feldspar,Calcite,Pressure,PhiC,Coord,Kf,RHOf,Sfact,Phi)%        [Phi,VpDry,VsDry,RHOBDry,VpSat,VsSat,RHOBSat]=softsediments(Clay,Feldspar,Calcite,Pressure,PhiC,Coord,Kf,RHOf,KClay,GClay,RhoClay,Sfact)                                                                    %% Calculates acoustic properties for the soft sediment model.  % Model uses Hertz-Mindlin contact theory to set the% right end member at critical porosity, and the  modified lower Hashin-Shtrikman % model to interpolate back to lower porosities.% The zero-porosity end point is pure mineral mix.% The high-porosity (PhiStart) end point comes from Hertz Mindlin Theory.% Assumes quartz/clay/feldspar/calcite solid phase%% Will prompt for inputs, if not specified)% INPUTS%   Clay			Volume clay content in solid phase (fraction)%   Feldspar		Volume feldspar content in solid phase (fraction)%   Calcite		    Volume calcite content in solid phase (fraction)%   Pressure		Effective pressure (MPa)%   PhiC			Critical porosity ~0.4%   Coord			Coordination number ~12%   Kf			    Pore fluid bulk modulus%   RHOf			Pore fluid density%   Kclay           Clay bulk modulus%   GClay           Clay shear modulus%   RhoClay         Clay bulk density%   Sfact           Empirical correction for shear modulus in Hertz-Mindlin calculation.%                   Defaults to 0.5%% OUTPUTS%   Phi			    Porosity (fraction) -- the same as in inputs%   VpDry			Dry rock Vp (km/s)%   VsDry			Dry rock Vs (km/s)%   RHOBDry			Dry rock Bulk density (g/cc)%   VpDry			Saturated rock Vp (km/s)%   VsDry			Saturated rock Vs (km/s)%   RHOBDry         Saturated rock bulk density (g/cc)%   Nu              Poisson ratio%Quartz content is 1-Clay-Feldspar-Calcite% Mineral component elastic moduli (K-Bulk,G-Shear/Modulus of rigidity)% Error factor introduced of 5% ---> scaled by 1.01 KQuartz,KQuartz=36.6e9; GQuartz=45e9; KClay=25e9; GClay=9e9; KFeldspar=75.6e9; GFeldspar=25.6e9;KCalcite=76.8e9; GCalcite=32.0e9;% Change just Quartz and ClayRhoQuartz, RHoClay% Mineral component densitiesRhoQuartz= 2650; RhoClay= 2580; RhoFeldspar= 2630; RhoCalcite=2710;if nargin<1    prompt = {'critical porosity',...             'Effective pressure (MPa)', ...             'Coordination Number', ... % Simple Cubic-6,Simple Hexagon-8,Dense random packing -9             'Clay Fraction', ...             'Calcite Fraction', ...             'Feldspar Fraction', ...             'Shear Reduction Factor', ...             'Fluid Bulk Modulus',...             'Fluid Density', ...             'Clay Bulk Modulus',...             'Clay Shear Modulus', ...             'Clay Bulk density'};        defans = {'.4', ... % Original is .4, change to 0.3             '10',...             '12', ...  % Original is 12, changed 10 - pore geometry              '.05',...  % Clay Fraction was 0.05, changed to 0.1             '0',...             '0', ...             '.5', ...             '2.5e9',...             '1020', ...             num2str(KClay), ...             num2str(GClay), ...             num2str(RhoClay)};    answer    = inputdlg(prompt,'Soft Sediment Model',1,defans);    PhiC      = str2num(answer{1});    Pressure  = str2num(answer{2});    Coord     = str2num(answer{3});    Clay      = str2num(answer{4});    Calcite   = str2num(answer{5});    Feldspar  = str2num(answer{6});    Sfact     = str2num(answer{7});    Kf        = str2num(answer{8});    RHOf      = str2num(answer{9});    KClay     = str2num(answer{10});    GClay     = str2num(answer{11});    RhoClay   = str2num(answer{12});end% Mineral phase moduli from Voigt-Reuss-Hill averageQuartz=1-Clay-Feldspar-Calcite;Ks=0.5.*(Quartz.*KQuartz+Clay.*KClay+Feldspar.*KFeldspar+Calcite.*KCalcite+1./(Quartz./KQuartz+Clay./KClay+Feldspar./KFeldspar+Calcite./KCalcite));Gs=0.5.*(Quartz.*GQuartz+Clay.*GClay+Feldspar.*GFeldspar+Calcite.*GCalcite+1./(Quartz./GQuartz+Clay./GClay+Feldspar./GFeldspar+Calcite./GCalcite));Ms=Ks+4.*Gs./3;NUs=0.5.*(Ms./Gs-2)./(Ms./Gs-1);% Mineral phase densityRHOs=Quartz.*RhoQuartz+Clay.*RhoClay+Feldspar.*RhoFeldspar+Calcite.*RhoCalcite;% Porosity runs between 0 and PhiC%Phi=[.000001:.001:PhiC+.000001]';RHOBDry=(1-Phi).*RHOs;P=Pressure.*1e6;C=Coord;% Effective K and G at PhiCKhat = ((C.^2.*(1-PhiC).^2.*Gs.^2.*P)./(18*pi^2*(1-NUs).^2)).^(1/3);Ghat = ((5-4*NUs)./(5*(2-NUs)))*((3*C.^2.*(1-PhiC).^2.*Gs.^2.*P)./(2*pi^2*(1-NUs).^2)).^(1/3);Ghat = Sfact*Ghat;% Lower HSKDry=1./((Phi./PhiC)./(Khat+4.*Ghat./3)+((PhiC-Phi)./PhiC)./(Ks+4.*Ghat./3))-4.*Ghat./3;ZZ1=(Ghat./6).*(9.*Khat+8.*Ghat)./(Khat+2.*Ghat);GDry=1./((Phi./PhiC)./(Ghat+ZZ1)+((PhiC-Phi)./PhiC)./(Gs+ZZ1))-ZZ1; MDry = KDry+(4./3).*GDry;NuDry=(MDry./GDry-2)./(MDry./GDry-1);% Fluid-saturated propertiesRHOBSat=RHOBDry+Phi.*RHOf;% Gassmannx = KDry./(Ks-KDry) + Kf./(Phi.*(Ks-Kf));KSat = Ks.*x./(1+x);% OutputsVpDry = sqrt((KDry+(4/3).*GDry)./RHOBDry);VsDry = sqrt(GDry./RHOBDry);VpSat = sqrt((KSat+(4/3).*GDry)./RHOBSat);VsSat = sqrt(GDry./RHOBSat);