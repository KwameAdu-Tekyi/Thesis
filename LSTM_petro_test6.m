%% Test 6:  Error In Assumed Rock Model
rng(42);
% View Original data
figure
subplot(151)
% To index into original data
corr_idx = length(XTrain)+ length(Xp_val) + test_idx; 

imagesc(1,yax,double(Facies(:,:,corr_idx)'));
title('Facies')

subplot(152)
plot(Yp_test(:,:,test_idx),yax), axis ij
title('Petro')

subplot(153)
plot(Elas(:,:,corr_idx),yax), axis ij
title('Elastic')

subplot(1,5,[4 5])
atldk(yax, ang,Xp_test(:,:,test_idx),'wiggle','accurate');
title('Seismog')
xlabel('Incidence Angles (°)')
axis tight

% Change Rock Physics Parameters and compute new input data
rng(42);

for i=1:length(Yp_test)
    fsim=hmmgenerate(length(yax),Tmat,Emat); % generate the facies profile using hidden Markov model
    temp=zeros(length(yax),3); %matrix for the petrophysical properties
    for j=1:size(Tmat,2) % cycle over the facies and distribute the petrophysical properties along the profile
        %generate random samples from the same multivariate normal distribution
        psim= mvnrnd(Mu_facies(:,j)',Cm_facies(:,:,j)); 
        psim=reshape(psim(:),[],3);
        [val,in]=find(fsim==j);
        temp(in,:)=psim(in,:);
    end
    
    %-- set the petrophysical properties within a feasible range
    %
    in=find(temp(:,1)<=0); %Phi
    temp(in,1)=eps;
    in=find(temp(:,1)>0.4);
    temp(in,1)=0.4; 
    %
    in=find(temp(:,2)<=0); % Sh
    temp(in,2)=eps;
    in=find(temp(:,2)>1);
    temp(in,2)=1;
    
    in=find(temp(:,3)<=0); % Sw
    temp(in,3)=eps;
    in=find(temp(:,3)>1);
    temp(in,3)=1;

    % Get Test Petrophysical Parameters
    [Vp_temp,Vs_temp,Rho_temp]=applico_RPM_new([temp(:,3) temp(:,1) temp(:,2)]); 
       
    % Compute the CMP with diff RPM (coord = 10) 
    d_pre6 = squeeze(calcola_dati_pre_stack(Vp_temp,Vs_temp,Rho_temp,repmat(wavelet2(:),1,length(ang)),ang));
    d_pre6 = reshape(d_pre6,[],length(ang));
       
    % store simulation
    noise6 = sqrt(0.05 .* var(d_pre6)).* randn(size(d_pre6)); % Adding the usual 5% noise
    d_pre6 = d_pre6 + noise6;
    Data_test6(:,:,i) = d_pre6; % CMP with new RPM
    Petro_test6(:,:,i) = temp; % Phi,Sh,Sw
    Elas_test6(:,:,i)  = [Vp_temp,Vs_temp,Rho_temp];
    Facies_test6(:,:,i) = categorical(fsim);
end

% Visualize Input Data for test 6
figure
subplot(151)
imagesc(1,yax,double(Facies_test6(:,:,test_idx)'))
ylabel('Time (s)')
xlabel('Facies')

subplot(152)
plot(Petro_test6(:,:,test_idx),yax), axis ij
xlabel('Petro')

subplot(153)
plot(Elas_test6(:,:,test_idx),yax), axis ij
xlabel('Elastic')

subplot(1,5,[4 5])
atldk(yax, ang,Data_test6(:,:,test_idx),'wiggle','accurate');
xlabel('Seismog')
axis tight

% Test on  Traing data for Test 6
Yp_predTest6 = minibatchpredict(net_petro,Data_test6,'MiniBatchSize',1);

% Evaluate errors
errors_NN(Yp_test, Yp_predTest6,test_idx,6);

% Visualize Results  of Test 6
plot_true_pred(Yp_test,Yp_predTest6,yax,test_idx,6);

% Avg RMS and % Err for entire test set
rmse_NN_test6 =  zeros(size(Yp_test,3),3);    % RMSE
pe_NN_test6 = zeros(size(Yp_test,3),3);       % Perc Error
r2_NN_test6 = zeros(size(Yp_test,3),3);       % R2 (coefficient of determination)

for i = 1:size(Yp_test,2)
    err = Yp_predTest6(:,:,i) - Yp_test(:,:,i);
    rmse_NN_test6(i,:) = sqrt(mean(err.^2));
    for j = 1 : size(pe_NN_test6,2)
        true = Yp_test(:,j,i);
        diff = Yp_predTest6(:,j,i) - true;
        pe_NN_test6(i,j) = 100*(sum(abs(diff(:)))./sum(abs(true(:))));
        xtemp = double(double(Yp_test(:,j,i)));
        ytemp = double(Yp_predTest6(:,j,i));
        [~,gof_petro] = fit(xtemp,ytemp,'poly1');
        r2_NN_test6(i,j) = gof_petro.rsquare;
     end
end

rmse_NN_test6 = mean(rmse_NN_test6); % Mean over all test samples
pe_NN_test6 = mean(pe_NN_test6); % Mean over all test samples
r2_NN_test6 = mean(r2_NN_test6);

sprintf('Avg RMSE error for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
    rmse_NN_test6(1),rmse_NN_test6(2),rmse_NN_test6(3))
sprintf('Avg Percentage error for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
    pe_NN_test6(1),pe_NN_test6(2),pe_NN_test6(3)) 
sprintf('Avg R2 for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
    r2_NN_test6(1),r2_NN_test6(2),r2_NN_test6(3)) 

% Data Error Test 6
data_test6 = zeros(size(Xp_test));
data_MAE_test6 = zeros(size(Xp_test,3),1);
data_PE_test6 = zeros(size(Xp_test,3),1);
data_MSE_test6 = zeros(size(Xp_test,3),1);
data_RMSE_test6 = zeros(size(Xp_test,3),1);
    
for i = 1: length(Yp_pred)
    petro_rpm = Yp_predTest6(:,:,i);
    % From petro to elastic
    [Vp_temp,Vs_temp,Rho_temp]=applico_RPM_new([petro_rpm(:,3) petro_rpm(:,1) petro_rpm(:,2)]); 
    % Compute the new CMP with pred petro properties
    d_pre_test6 = (calcola_dati_pre_stack(Vp_temp,Vs_temp,Rho_temp,repmat(wavelet(:),1,length(ang)),ang));
    data_test6(:,:,i) = reshape(d_pre,[],length(ang));
      
    % Compute data error
    true_data = Xp_test(:,:,i); % compare with the actual data
    data_diff = data_test6(:,:,i) - true_data ; % error
    data_MAE_test6(i) = mean(abs(data_diff(:)));
    data_PE_test6(i) = 100*(sum(abs(data_diff(:)))./sum(abs(true_data(:))));
    data_MSE_test6(i) = mean(data_diff(:).^2);
    data_RMSE_test6(i) = sqrt(data_MSE_test6(i));
end

sprintf('Data Percentage Error for test sample: %.2f', data_PE_test6(test_idx))

% Average data error
data_mae_NN_test6 = mean(data_MAE_test6);   % Data mean squared error
data_pe_NN_test6 = mean(data_PE_test6);     % Data mean error
data_mse_NN_test6 = mean(data_MSE_test6);   % Mean over all test samples
data_rmse_NN_test6 = mean(data_RMSE_test6); % Mean over all test samples

sprintf('Avg Data Errors: ')
sprintf('Avg Data MAE  : %.2f',data_mae_NN_test6)
sprintf('Avg Data PE   : %.2f',data_pe_NN_test6)
sprintf('Avg Data MSE  : %.2f',data_mse_NN_test6)
sprintf('Avg Data RMSE : %.2f',data_rmse_NN_test6)

% Compare the actual and computed seismograms
figure; 
atldk(yax,ang,data_test6(:,:,test_idx),'color','r'); % pred
xlabel('Angle (°)');ylabel('Time (s)');
axis tight
hold on
%atldk(yax,ang,Data_test6(:,:, test_idx),'color','k'); % true
atldk(yax,ang,Xp_test(:,:,test_idx),'color','k');
xlabel('Angle (°)');ylabel('Time');  
title('True vs Predicted Seismogram')
axis tight       
hold off

% Compare Elastic Properties
figure;
plot(Elas(:,:,corr_idx),yax,'LineWidth',1.5); axis ij 
hold on
plot(Elas_test6(:,:,test_idx),yax,'k--',LineWidth = 1.5); axis ij
xlabel('Elastic Properties')
ylabel('Time (s)')
title('True vs Predicted Elastic Properties')
grid on
legend('Vp','Vs','Rho','Pred','Location','best')

% R2 fit :Scatter plot & Regression fit
% Vp
compute_Rsquared(Elas(:,1,corr_idx), Elas_test6(:,1,test_idx),'Vp');
% Vs
compute_Rsquared(Elas(:,2,corr_idx) ,Elas_test6(:,2,test_idx),'Vs');
% Rho
compute_Rsquared(Elas(:,3,corr_idx), Elas_test6(:,3,test_idx),'Rho');