%% Test 5:  Error In Assumed Prior Elastic and Facies Models
rng(42);
% a)Tweak the transition matrix to increase shale probability by 5%
%Tmat_test5 = Tmat;
Tmat_test5 = [
    0.95 0.025 0.025;...
    0.15 0.85 0;...
    0.15 0.05 0.8];

%% NB: PROBABLY TOO LARGE PERTURBATIONS: start from 1.01 and increase gradually
% b)Tweak covariance matrix to in increase errors to 5%
Cm_facies_test5 = Cm_facies;
Cm_facies_test5 = 1.05 .* Cm_facies;

% c)Tweak mean
Mu_facies_test5 = 1.05.*Mu_facies;
%Mu_facies_test5= Mu_facies;

% d) Combine a,b and c
% Generate seismograms to honour new priors
rng(42);

for i=1:length(Xp_test)
    fsim=hmmgenerate(length(yax),Tmat_test5,Emat); %generate the facies profile using hidden Markov model
    temp=zeros(length(yax),3); %matrix for the petrophysical properties
    for j=1:size(Tmat_test5,2) % cycle over the facies and distribute the petrophysical properties along the profile
        %generate random samples from the same multivariate normal distribution
        psim= mvnrnd(Mu_facies_test5(:,j)',Cm_facies_test5(:,:,j));  % tweak mean and covariances ()
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
    
    % apply the rock physics model  % try diff rock physics models 
    [Vp_temp,Vs_temp,Rho_temp]=applico_RPM_new([temp(:,3) temp(:,1) temp(:,2)]); 
       
    % compute the CMP
    d_pre5=squeeze(calcola_dati_pre_stack(Vp_temp,Vs_temp,Rho_temp,repmat(wavelet(:),1,length(ang)),ang));
    d_pre5=reshape(d_pre5,[],length(ang));
    noise5 = sqrt(0.05 .* var(d_pre5)).* randn(size(d_pre5)); % Adding the usual 5% noise
    d_pre5 = d_pre5 + noise5;
    
    % store simulation
    Data_test5(:,:,i)= d_pre5;
    Petro_test5(:,:,i)= temp; % Phi,Sh,Sw
    Elas_test5(:,:,i)= [Vp_temp,Vs_temp,Rho_temp];
    Facies_test5(:,:,i)= categorical(fsim);    
end

% Display test data sample
% figure; 
% atldk(yax,ang,Data_test5{test_idx},'wiggle','accurate','color','k') % pred
% xlabel('Angle (°)');ylabel('Time (s)');
% axis tight

figure
subplot(151)
imagesc(1,yax,double(Facies_test5(:,:,test_idx)'))
title('Facies')
ylabel('Time (s)')

subplot(152)
plot(Petro_test5(:,:,test_idx),yax), axis ij
title('Petro')

subplot(153)
plot(Elas_test5(:,:,test_idx),yax), axis ij
title('Elastic')

subplot(1,5,[4 5])
atldk(yax, ang,Data_test5(:,:,test_idx),'wiggle','accurate','color','k');
title('Seismog')
xlabel('Incidence Angles (°)')
axis tight

% Test Neural Net of newly generated data for Test 5
Yp_predTest5 = minibatchpredict(net_petro,Data_test5,'MiniBatchSize',1);
%Yp_predTest5 = minibatchpredict(netPetroTuned,Data_test5,'MiniBatchSize',1);

% Evaluate the errors for test sample
% test_idx =   393;
errors_NN(Yp_test, Yp_predTest5, test_idx,5); 
    
% Visualize predictions : Predicted vs True Petrophysical properties
plot_true_pred(Yp_test,Yp_predTest5,yax,test_idx,5); 

% Avg RMS and PE Err for all test set
rmse_NN_test5 =  zeros(size(Yp_test,3),3);    % RMSE
pe_NN_test5 = zeros(size(Yp_test,3),3);       % Perc Error
r2_NN_test5 = zeros(size(Yp_test,3),3);       % R2 (coefficient of determination)

for i = 1:size(Yp_test,3)
    err = Yp_predTest5(:,:,i) - Yp_test(:,:,i);
    rmse_NN_test5(i,:) = sqrt(mean(err.^2));
    for j = 1 : size(pe_NN,2)
        true_data = Yp_test(:,j,i);
        diff = Yp_predTest5(:,j,i)- true_data ;
        pe_NN_test5(i,j) = 100*(sum(abs(diff(:)))./sum(abs(true_data(:))));
        xtemp = double(Yp_test(:,j,i));
        ytemp = double(Yp_predTest5(:,j,i));
        [~,gof_petro] = fit(xtemp,ytemp,'poly1');
        r2_NN_test5(i,j) = gof_petro.rsquare;
    end
end

rmse_NN_test5 = mean(rmse_NN_test5); % Mean over all test samples
pe_NN_test5 = mean(pe_NN_test5); % Mean over all test samples
r2_NN_test5 = mean(r2_NN_test5);

sprintf('Avg RMSE error for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
    rmse_NN_test5(1),rmse_NN_test5(2),rmse_NN_test5(3))
sprintf('Avg Percentage error for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
    pe_NN_test5(1),pe_NN_test5(2),pe_NN_test5(3))
sprintf('Avg R2 for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
    r2_NN_test5(1),r2_NN_test5(2),r2_NN_test5(3))
 
% Data Error for Test 5
data_test5 = zeros(size(Xp_test));
data_MAE_test5 = zeros(size(Xp_test,3),1);
data_PE_test5 = zeros(size(Xp_test,3),1);
data_MSE_test5 = zeros(size(Xp_test,3),1);
data_RMSE_test5 = zeros(size(Xp_test,3),1);
    
for i = 1: length(Yp_pred)
    petro_rpm = Yp_predTest5(:,:,i);
    % From petro to elastic
    [Vp_temp,Vs_temp,Rho_temp]=applico_RPM_new([petro_rpm(:,3) petro_rpm(:,1) petro_rpm(:,2)]); 
    % Compute the new CMP with pred petro properties
    d_pre = (calcola_dati_pre_stack(Vp_temp,Vs_temp,Rho_temp,repmat(wavelet(:),1,length(ang)),ang));
    data_test5(:,:,i) = reshape(d_pre,[],length(ang));
      
    % Compute data error
    true_data = Xp_test(:,:,i); % compare with the actual data
    data_diff = true_data - data_test5(:,:,i) ; % error
    data_MAE_test5(i) = mean(abs(data_diff(:)));
    data_PE_test5(i) = 100*(sum(abs(data_diff(:)))./sum(abs(true_data(:))));
    data_MSE_test5(i) = mean(data_diff(:).^2);
    data_RMSE_test5(i) = sqrt(data_MSE_test5(i));
end

sprintf('Data Percentage Error for test sample: %.2f', data_PE_test5(test_idx)) % Average data error
data_mae_NN_test5 = mean(data_MAE_test5);   % Data mean squared error
data_pe_NN_test5 = mean(data_PE_test5);     % Data mean error
data_mse_NN_test5 = mean(data_MSE_test5);   % Mean over all test samples
data_rmse_NN_test5 = mean(data_RMSE_test5); % Mean over all test samples
sprintf('Avg Data Errors: ')
sprintf('Avg Data MAE  : %.2f',data_mae_NN_test5)
sprintf('Avg Data PE   : %.2f',data_pe_NN_test5)
sprintf('Avg Data MSE  : %.2f',data_mse_NN_test5)
sprintf('Avg Data RMSE : %.2f',data_rmse_NN_test5)

%%ù Compare two seismograms
figure; 
atldk(yax,ang,data_test5(:,:,test_idx),'color','r', 'linestyle','-') ;% pred
xlabel('Angle (°)');ylabel('Time (s)');
axis tight
hold on
atldk(yax,ang,Xp_test(:,:,test_idx),'color','k'); % true
xlabel('Angle (°)');ylabel('Time');  
title('True vs Predicted Seismogram')
axis tight       
hold off