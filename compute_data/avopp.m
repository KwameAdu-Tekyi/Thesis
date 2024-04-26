function [Rpp,A,B]=avopp(vp_arg,vs_arg,rho_arg,vp2,vs2,d2,ang,approx)

%function Rpp=avopp(vp_arg,vs_arg,rho_arg,vp2,vs2,d2,ang,approx);
%AVOPP AVO Rpp vs. angle
%Calculates P-to-P reflectivity (Rpp) as a function of
%the angle of incidence (ang).
%input parameters:
%  layer 1 (top): vp_arg, vs_arg, density1 (rho_arg)
%  layer 2 (bottom): vp2, vs2, density2 (d2)
% ang: vector with angles(DEG)
% approx:  1)Full Zoeppritz(A&R)
%	       2)Aki&Richards
%          3)Shuey's paper
%          4)Castagna's paper->Shuey (slightly different formulation of Shuey)
%
% With no output arguments, plots Rpp vs. angle.
%
% See also AVOPS, AVO_ABE

% written by Ezequiel Gonzalez (Oct,1999)

t=ang.*pi./180;	p=sin(t)./vp_arg;	ct=cos(t);
da=(rho_arg+d2)/2;     
Dd=(d2-rho_arg);
vpa=(vp_arg+vp2)/2;  Dvp=(vp2-vp_arg);
vsa=(vs_arg+vs2)/2;  Dvs=(vs2-vs_arg);

switch approx
   case 1		%FULL Zoeppritz (A&K)
	ct2=sqrt(1-(sin(t).^2.*(vp2.^2./vp_arg.^2)));
	cj1=sqrt(1-(sin(t).^2.*(vs_arg.^2./vp_arg.^2)));
	cj2=sqrt(1-(sin(t).^2.*(vs2.^2./vp_arg.^2)));
	a=(d2.*(1-(2.*vs2.^2.*p.^2)))-(rho_arg.*(1-(2.*vs_arg.^2.*p.^2)));
	b=(d2.*(1-(2.*vs2.^2.*p.^2)))+(2.*rho_arg.*vs_arg.^2.*p.^2);
	c=(rho_arg.*(1-(2.*vs_arg.^2.*p.^2)))+(2.*d2.*vs2.^2.*p.^2);
	d=2.*((d2.*vs2.^2)-(rho_arg.*vs_arg.^2));
	E=(b.*ct./vp_arg)+(c.*ct2./vp2);
	F=(b.*cj1./vs_arg)+(c.*cj2./vs2);
	G=a-(d.*ct.*cj2./(vp_arg.*vs2));
	H=a-(d.*ct2.*cj1./(vp2.*vs_arg));
	D=(E.*F)+(G.*H.*p.^2);
	Rpp=( (((b.*ct./vp_arg)-(c.*ct2./vp2)).*F) -  ((a+(d.*ct.*cj2./(vp_arg.*vs2))).*H.*p.^2) ) ./ D;
   case 2		%Aki & Richard (aprox)
    %assuming (angles) i=i1
	Rpp=(0.5.*(1-(4.*p.^2.*vsa.^2)).*Dd./da) + (Dvp./(2.*ct.^2.*vpa)) - (4.*p.^2.*vsa.*Dvs);
   case 3		%Shuey
	poi1=((0.5.*(vp_arg./vs_arg).^2)-1)./((vp_arg./vs_arg).^2-1);
	poi2=((0.5.*(vp2./vs2).^2)-1)./((vp2./vs2).^2-1);
	poia=(poi1+poi2)./2;   Dpoi=(poi2-poi1);
	Ro=0.5.*((Dvp./vpa)+(Dd./da));
	Bx=(Dvp./vpa)./((Dvp./vpa)+(Dd./da));
	Ax=Bx-(2.*(1+Bx).*(1-2.*poia)./(1-poia));
	Rpp= Ro + (((Ax.*Ro)+(Dpoi./(1-poia).^2)).*sin(t).^2) + ...
	         (0.5.*Dvp.*(tan(t).^2-sin(t).^2)./vpa);

   case 4		%Shuey linear
	A=0.5.*((Dvp./vpa)+(Dd./da));
	B=(-2.*vsa.^2.*Dd./(vpa.^2.*da)) + (0.5.*Dvp./vpa) - (4.*vsa.*Dvs./(vpa.^2));
	Rpp=A + (B.*sin(t).^2);
    %case 5 %Hilterman and Graul(2009)
    %Eqn 11 pg 19: Calculation of Synthetic gather by  Graham Ganssle
    otherwise
    
end

if nargout==0 % plots if no variable assignment is made at function call
    plot(ang,Rpp)
end
