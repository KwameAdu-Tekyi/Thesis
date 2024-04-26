function [w] = ricker_Sacchi(f,dt)
%RICKER: Ricker wavelet
%
%  This function designs a Ricker wavelet with
%  peak frequence f.
%
%  [w] = ricker(f,dt);
%
%  IN    f: central freq. in Hz (f <<1/(2dt))
%        dt: sampling interval in sec 
%
%  OUT   w: the ricker wavelet
%
%
%  Example:
%            w = ricker(40,0.004); plot(w);
%           
%  SeismicLab
%  Version 1
%
%  written by M.D.Sacchi, last modified December 10,  1998.
%  sacchi@phys.ualberta.ca
%
%  Copyright (C) 1998 Seismic Processing and Imaging Group
%                     Department of Physics
%                     The University of Alberta
%
%

nw=6./f/dt;
nw=2*floor(nw/2)+1;
nc=floor(nw/2);
i=1:nw;
alpha=(nc-i+1).*f*dt*pi;
beta=alpha.^2;
w=(1.-beta.*2).*exp(-beta);
