%% ----------------------------Test 3--------------------------------------
% Error in noise estimation---> 20% of the std of test data
rng(42);

Xp_test3 = zeros(size(Xp_test));
var_level = 0.2; % 20 percent 

for c = 1 : size(Xp_test,3)
    % Variance of XTest 
    desired_variance  = var_level .* var(Xp_test(:,:,c)); % 10 percent of the variance of test 1 data
    noise_Xp = sqrt(desired_variance).* randn(size(Xp_test(:,:,c))); % scaled noise
    % Add noise to test seismogram, Xp_test
    Xp_test3(:,:,c) =  Xp_test(:,:,c) + noise_Xp;
end

% Visualize noisy Seismogram
figure
atldk(yax,ang,Xp_test3(:,:,test_idx),'color','r'); % Noisy 
hold on 
atldk(yax,ang,Xp_test(:,:,test_idx),'color','k'); % true
hold off
title('Noisy vs Original Test Seismogram')
xlabel('Incidence Angle (°)')
%xlim([-10 50])
axis tight

% Petrophysical LSTM predictions on noisy data
Yp_predTest3 = minibatchpredict(net_petro,Xp_test3,'MiniBatchSize',1);

% Evaluate Predictions
errors_NN(Yp_test, Yp_predTest3,test_idx,3);

% Visualize predictions : Predicted vs True Petrophysical properties
plot_true_pred(Yp_test,Yp_predTest3,yax,test_idx,3);

% Avg RMS and % Err for entire test set
rmse_NN_test3 =  zeros(size(Yp_test,3),3);    % RMSE
pe_NN_test3 = zeros(size(Yp_test,3),3);       % Perc Error
r2_NN_test3 = zeros(size(Yp_test,3),3);

for i = 1:size(Yp_test,3)
    err = Yp_predTest3(:,:,i) - Yp_test(:,:,i); 
    rmse_NN_test3(i,:) = sqrt(mean(err.^2));
    for j = 1 : size(pe_NN_test3,2)
        diff = Yp_predTest3(:,j,i) - Yp_test(:,j,i);
        pe_NN_test3(i,j) = 100*(sum(abs(diff)) ./sum(abs(Yp_test(:,j,i))));
        xtemp = double(Yp_test(:,j,i));
        ytemp = double(Yp_predTest3(:,j,i));
        [~,gof_petro] = fit(xtemp,ytemp,'poly1');
        r2_NN_test3(i,j) = gof_petro.rsquare;
     end
end
rmse_NN_test3 = mean(rmse_NN_test3);        % Mean over all test samples
pe_NN_test3 = mean(pe_NN_test3);            % Mean over all test samples
r2_NN_test3 = mean(r2_NN_test3);            % Mean over all test samples

sprintf('Avg RMSE error for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
    rmse_NN_test3(1),rmse_NN_test3(2),rmse_NN_test3(3))
sprintf('Avg Percentage error for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
    pe_NN_test3(1),pe_NN_test3(2),pe_NN_test3(3))
sprintf('Avg R2 for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
    r2_NN_test3(1),r2_NN_test3(2),r2_NN_test3(3))

% Data Error for Test2
data_test3 = zeros(size(Xp_test));
data_MAE_test3 = zeros(size(Xp_test,3),1);
data_PE_test3 = zeros(size(Xp_test,3),1);
data_MSE_test3 = zeros(size(Xp_test,3),1);
data_RMSE_test3 = zeros(size(Xp_test,3),1);

for i = 1: size(Yp_pred,3)
    petro_rpm = Yp_predTest3(:,:,i);
    % From petro to elastic
    [Vp_temp,Vs_temp,Rho_temp]=applico_RPM_new([petro_rpm(:,3) petro_rpm(:,1) petro_rpm(:,2)]); 
    
    % Compute the new CMP with pred petro properties
    d_pre = (calcola_dati_pre_stack(Vp_temp,Vs_temp,Rho_temp,repmat(wavelet(:),1,length(ang)),ang));
    data_test3(:,:,i) = reshape(d_pre,[],length(ang));
    
    % Compute data error
    true_data = Xp_test(:,:,i);
    data_diff = data_test3(:,:,i)  - true_data ; % error/ difference
    data_MAE_test3(i) = mean(abs(data_diff(:)));
    data_PE_test3(i) = 100*(sum(abs(data_diff(:)))./sum(abs(true_data(:))));
    data_MSE_test3(i) = mean(data_diff(:).^2);
    data_RMSE_test3(i) = sqrt(data_MSE_test3(i));
end

sprintf('Data Percentage Error for test sample: %.2f', data_PE_test3(test_idx)) % Average data error

% Average data error
sprintf('Avg Data Error: ')
sprintf('Avg Data MAE:  %.3f', round(mean(data_MAE_test3,'omitnan'),4))   % Data mean squared error
sprintf('Avg Data PE:   %.3f', round(mean(data_PE_test3, 'omitnan'),4))   % Mean over all test samples
sprintf('Avg Data MSE:  %.3f', round(mean(data_MSE_test3, 'omitnan'),4))  % Mean over all test samples
sprintf('Avg Data RMSE: %.3f', round(mean(data_RMSE_test3, 'omitnan'),4)) % Mean over all test samples

% Compare two seismograms
figure; atldk(yax,ang,data_test3(:,:,test_idx),'color','r'); % pred
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
