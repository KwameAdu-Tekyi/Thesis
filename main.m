clear all
close all


addpath(genpath(pwd))
load modello_sharpen2 % a reference petrophysical model from which we derive the prior petrophysical assumptions

% cut off values for the tree litho fluid facies
in_shale=find(Phi_true<0.1);
in_brine_sand=find(Phi_true>0.1 & Sw_true>0.5);
in_gas_sand=find(Phi_true>0.1 & Sw_true<=0.5);

in_tot{1}=in_shale;
in_tot{2}=in_brine_sand;
in_tot{3}=in_gas_sand;

% definition of a Gaussian mixture prior model. Gaussian assumption within
% each litho fluid class
for i=1:size(in_tot,2)
    Mu(i,:)=mean([Phi_true(in_tot{i}) Sh_true(in_tot{i}) Sw_true(in_tot{i})]);
    Sigma(:,:,i)=cov([Phi_true(in_tot{i}) Sh_true(in_tot{i}) Sw_true(in_tot{i})]);
end

% definition of the vertical transition matrix for the three classes (you can also treat this matrix as a random variable)
Tmat=[0.9 0.05 0.05;...
     0.1 0.9 0;...
      0.1 0.1 0.8];
Emat=eye(size(in_tot,2));

%% Generation of 1D petrophysical models honouriung the prior assumptions

rng(42);
DY=0.002; %sampling rate (s)
yax=0:DY:(100)*DY; % time axis
tau_y=-(DY)*(length(yax)-1):(DY):(DY)*(length(yax)-1); % correlation axis
ni_tau_y=exp(-(tau_y/0.005).^2); % vertical correlation for the petrophysical properties. To impose vertical continuity to the generated 1D petrophysical profiles
NI_TAU_Y=toeplitz(ni_tau_y);
NI_TAU_Y=NI_TAU_Y(length(yax):end,1:length(yax));

% prior mean and covariance (the last also including the vertical
% correlation)
for i=1:size(in_tot,2)
    Cm_facies(:,:,i)=(kron(Sigma(:,:,i),NI_TAU_Y));
    Mu_facies(:,i)=reshape(repmat(Mu(i,:),length(yax),1),[],1);
end

Nsim = 10000; % number of prior simulations (just an example)
wavelet=ricker_Sacchi(50,DY); % source wavelet
ang=0:10:40; % incidence angles to consider

for i=1:Nsim
    fsim=hmmgenerate(length(yax),Tmat,Emat); % generate the facies profile using hidden Markov model
    temp=zeros(length(yax),3); % matrix for the petrophysical properties
    for j=1:size(Tmat,2) % cicle over the facies and distribute the petrophysical properties along the profile
        psim=mvnrnd(Mu_facies(:,j)',Cm_facies(:,:,j));
        psim=reshape(psim(:),[],3);
        [val,in]=find(fsim==j);
        temp(in,:)=psim(in,:);
    end
    
    %-- set the petrophysical properties within a feasible range
    in=find(temp(:,1)<=0);
    temp(in,1)=eps;
    in=find(temp(:,1)>0.4);
    temp(in,1)=0.4;
    
    in=find(temp(:,2)<=0);
    temp(in,2)=eps;
    in=find(temp(:,2)>1);
    temp(in,2)=1;
    
    in=find(temp(:,3)<=0);
    temp(in,3)=eps;
    in=find(temp(:,3)>1);
    temp(in,3)=1;
    %----
    
    % apply the rock physics model    
    [Vp_temp,Vs_temp,Rho_temp]=applico_RPM_new([temp(:,3) temp(:,1) temp(:,2)]); 
       
    % compute the CMP
    d_pre=squeeze(calcola_dati_pre_stack(Vp_temp,Vs_temp,Rho_temp,repmat(wavelet(:),1,length(ang)),ang));
    d_pre=reshape(d_pre,[],length(ang));
    
    % store simulation
    Data(:,:,i)=d_pre;
    Petro(:,:,i)=temp;
    Elas(:,:,i)=[Vp_temp,Vs_temp,Rho_temp];
    Facies(:,:,i)=categorical(fsim);    
end


%%
%test_idx = randi(length(Yp_test),1);; % index of the model to represent
figure
subplot(151)
imagesc(1,yax,double(Facies(:,:,test_idx)'))
ylabel('Time (s)')
title('Facies')

subplot(152)
plot(Petro(:,:,test_idx),yax), axis ij
title('Petro')

subplot(153)
plot(Elas(:,:,test_idx),yax), axis ij
title('Elastic')

subplot(1,5,[4 5])
atldk(yax, ang,Data(:,:,test_idx),'wiggle','accurate');
xlabel('Angle (°)')
title('Syn. Seismog.')

% Shale, low-porosity sand and high-porosity sand (SH -1, LPS-2 HPS-3)
% Plotting the distrution of Facies-dependent components of the Petrophysical prior models for Phi,Sh and Sw.

pdf_PetroParams(Facies,Petro,test_idx,1)
xlabel('Phi')
xlim([-0.04 0.35])
pdf_PetroParams(Facies,Petro,test_idx,2)
xlabel('Sh')
pdf_PetroParams(Facies,Petro,test_idx,3)
xlabel('Sw')
xlim([0 1.1])