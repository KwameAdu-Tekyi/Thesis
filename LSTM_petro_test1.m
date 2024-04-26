%% -------------------------------- Test 1 ------------------------------------- 
% Testing the network (same noise statistics in training and test data) 
rng(42)
Yp_pred = minibatchpredict(net_petro,Xp_test,'MiniBatchSize',1);
%Yp_pred = predict(netPetroTuned,Xp_test,'MiniBatchSize',1);

% Evaluate Predictions
errors_NN(Yp_test, Yp_pred,test_idx,1);

% Visualize predictions : Predicted vs True Petrophysical properties
plot_true_pred(Yp_test,Yp_pred,yax,test_idx,1);

% Avg RMS and % Err for entire test set
rmse_NN_test1 =  zeros(size(Yp_test,3),3);    % RMSE
pe_NN_test1 = zeros(size(Yp_test,3),3);       % Perc Error
r2_NN_test1 = zeros(size(Yp_test,3),3);       % R2 - coeff of determination

for i = 1:size(Yp_test,3)
    err = Yp_pred(:,:,i) - Yp_test(:,:,i) ;
    rmse_NN_test1(i,:) = sqrt(mean(err.^2));
    for j = 1 : size(pe_NN_test1,2)
        true = Yp_test(:,j,i);
        diff = Yp_pred(:,j,i) - true;
        pe_NN_test1(i,j) = 100*(sum(abs(diff))./sum(abs(true)));
        xtemp = double(Yp_test(:,j,i));
        ytemp = double(Yp_pred(:,j,i));
        [~,gof_petro] = fit(xtemp ,ytemp,'poly1');
        r2_NN_test1(i,j) = gof_petro.rsquare;
    end
end

rmse_NN_test1 = mean(rmse_NN_test1);     % Mean over all test samples
pe_NN_test1 = mean(pe_NN_test1);         % Mean over all test samples
r2_NN_test1 = mean(r2_NN_test1);         % Mean R2 over all test samples

sprintf('Avg RMSE error for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
    rmse_NN_test1(1),rmse_NN_test1(2),rmse_NN_test1(3))
sprintf('Avg Percentage error for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
    pe_NN_test1(1),pe_NN_test1(2),pe_NN_test1(3))
sprintf('Avg R2 for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
    r2_NN_test1(1),r2_NN_test1(2),r2_NN_test1(3))


% Data Error for Test 1
data_test1 = zeros(size(Xp_test));
data_MAE_test1 = zeros(size(Xp_test,3),1);
data_PE_test1 = zeros(size(Xp_test,3),1);
data_MSE_test1 = zeros(size(Xp_test,3),1);
data_RMSE_test1 = zeros(size(Xp_test,3),1);

for i = 1: size(Yp_pred,3)
    petro_rpm = Yp_pred(:,:,i);
    % From petro to elastic
    [Vp_temp,Vs_temp,Rho_temp]=applico_RPM_new([petro_rpm(:,3) petro_rpm(:,1) petro_rpm(:,2)]); 
    
    % Compute the new CMP with pred petro properties
    d_pre = (calcola_dati_pre_stack(Vp_temp,Vs_temp,Rho_temp,repmat(wavelet(:),1,length(ang)),ang));
    data_test1(:,:,i) = reshape(d_pre,[],length(ang));
    
    % Compute data error
    true_data = Xp_test(:,:,i);
    data_diff = true_data -  data_test1(:,:,i) ; % error/ difference
    data_MAE_test1(i) = mean(abs(data_diff(:)));
    data_PE_test1(i) = 100*(sum(abs(data_diff(:)))./sum(abs(true_data(:))));
    data_MSE_test1(i) = mean(data_diff(:).^2);
    data_RMSE_test1(i) = sqrt(data_MSE_test1(i));
  
end

sprintf('Data Percentage Error for test sample: %.2f', data_PE_test1(test_idx)) % Average data error

% Average data error
sprintf('Avg Data Error: ')
sprintf('Avg Data MAE:  %.3f', round(mean(data_MAE_test1,'omitnan'),4))   % Data mean squared error
sprintf('Avg Data PE:   %.3f', round(mean(data_PE_test1, 'omitnan'),4))   % Mean over all test samples
sprintf('Avg Data MSE:  %.3f', round(mean(data_MSE_test1, 'omitnan'),4))  % Mean over all test samples
sprintf('Avg Data RMSE: %.3f', round(mean(data_RMSE_test1, 'omitnan'),4)) % Mean over all test samples

% Compare two seismograms
figure; atldk(yax,ang,data_test1(:,:,test_idx),'color','r'); % pred
xlabel('Angle (°)');ylabel('Time (s)');
%title('Seismogram from predicted elastic')
axis tight
hold on
%figure; 
atldk(yax,ang,Xp_test(:,:,test_idx),'color','k'); % true
xlabel('Angle (°)');ylabel('Time'); 
title('True vs Predicted Seismogram')
axis tight       
hold off