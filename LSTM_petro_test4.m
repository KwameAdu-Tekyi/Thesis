%% Test 4: Erroneous Peak Frequency and phase of source wavelet
rng(42);
Xp_test_4 = zeros(size(Xp_test));
f0 = 45;                % change dominant frequency from 50 to 45 and 40
dt = DY;                % time step
wavelet2 = ricker_Sacchi(f0,DY); %source wavelet
wavelet2 = 0.8*wavelet2; % Reduced amplitude from 1 to 0.8
l = length(wavelet2);    % No of samples

if rem(l,2) == 0
    wavelet2 = [wavelet2; 0];
    l = length(wavelet2);
end

l_mid = (l-1)/2;
t = -(l_mid)*dt:dt:l_mid*dt;   % time vector

%%Original wavelet
w = zeros(1,length(wavelet2));
w(8:68) = wavelet;
% figure; plot(t,w);axis tight

% Amplitude spectrum 
W = fft(wavelet2(:));
AW = abs(W);
PW = angle(W);
tax = (0:l-1)*dt;
ax = f0 * linspace(-dt,dt,l)';
%tax=ax/f0;

phase_shift = [pi/12 pi/6 ] ; % Phase shift of 15° and 30°
%freq_new = [40 45]; % New frequencies
phaseInDegrees = [15 ,30];
rng(42);

% Loop for Seismogram of diff source wavelet
for p = 1: length(phase_shift)
    
    % Computing new wavelet (phase shifted wavelet)
    W_new = zeros(size(W));
    W_new(2:l_mid) = W(2:l_mid).*exp(+1i*phase_shift(p));
    W_new(l_mid+2:end) = W(l_mid+2:end).*exp(-1i*phase_shift(1));
    w_new = real(ifft(W_new));
    % figure;
    % plot(tax,w,'b',tax,w_new,'r','LineWidth',1.6)
    % xlabel('Time (s)');ylabel('Amplitude')
    % title(['Phase Shift: ', num2str(phaseInDegrees(p)),' °'])
    % legend('Original wavelet','Erroneous wavelet')
    
    % Looping over the Vp,Vs,Rho to compute the new sesismograms
    for i = 1:length(Ye_test)
        
        % Get the elastic parameters
        Vp_temp = Ye_test(:,1,i);
        Vs_temp = Ye_test(:,2,i);
        Rho_temp= Ye_test(:,3,i);
        
        % Recompute the CMP gathers with (erroneous wavelet and elastics)
        data_errWavelet = squeeze(calcola_dati_pre_stack(Vp_temp,Vs_temp,Rho_temp,...
        repmat(w_new,1,length(ang)),ang));
        data_errWavelet = reshape(data_errWavelet,[],length(ang));
        noise_level = 0.05;
        noise = sqrt(noise_level .* var(data_errWavelet)).* randn(size(data_errWavelet));
        Xp_test_4(:,:,i) = data_errWavelet + noise;
    end
    
    % Predict with the neural network
    %Yp_predTest4 = predict(net_petro,Xp_test_4,'MiniBatchSize',1);
    Yp_predTest4 = minibatchpredict(net_petro,Xp_test_4,'MiniBatchSize',1);

    % Evaluate Errors
    %clc
    sprintf('RESULTS FROM TEST 4 at %d°' ,phaseInDegrees(p))
    errors_NN(Yp_test, Yp_predTest4,test_idx,4);
    
    % Visualize predictions : Predicted vs True Petrophysical properties
    plot_true_pred(Yp_test,Yp_predTest4,yax,test_idx,4) ;
    
    % Avg RMS and % Err for entire test set
    rmse_NN_test4 =  zeros(size(Yp_test,3),3);    % RMSE
    pe_NN_test4 = zeros(size(Yp_test,3),3);       % Perc Error
    r2_NN_test4 = zeros(size(Yp_test,3),3);       % R2

    for i = 1:size(Yp_test,3)
        err = Yp_predTest4(:,:,i) - Yp_test(:,:,i);
        rmse_NN_test4(i,:) = sqrt(mean(err.^2));
        for j = 1 : size(pe_NN_test4,2)
            diff = Yp_predTest4(:,j,i) - Yp_test(:,j,i);
            pe_NN_test4(i,j) = 100*(sum(abs(diff)) ./sum(abs(Yp_test(:,j,i))));
            xtemp = double(Yp_test(:,j,i));
            ytemp = double(Yp_predTest4(:,j,i));
            [~,gof_petro] = fit(xtemp,ytemp,'poly1');
            r2_NN_test4(i,j) = gof_petro.rsquare;
        end
    end
    % Mean of respective error metrics of test data
    rmse_NN_test4 = mean(rmse_NN_test4); % Mean over all test samples
    pe_NN_test4 = mean(pe_NN_test4); % Mean over all test samples
    r2_NN_test4 = mean(r2_NN_test4);
    sprintf('Avg RMSE error for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
        rmse_NN_test4(1),rmse_NN_test4(2),rmse_NN_test4(3))
    sprintf('Avg Percentage error for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
        pe_NN_test4(1),pe_NN_test4(2),pe_NN_test4(3))
    sprintf('Avg R2 for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
        r2_NN_test4(1),r2_NN_test4(2),r2_NN_test4(3))
    
    
    % Data Error for Test 4
    data_test4= zeros(size(Xp_test));
    data_MAE_test4 = zeros(size(Xp_test,3),1);
    data_PE_test4 = zeros(size(Xp_test,3),1);
    data_MSE_test4 = zeros(size(Xp_test,3),1);
    data_RMSE_test4 = zeros(size(Xp_test,3),1);
        
    for i = 1: length(Yp_pred)
        petro_rpm = Yp_predTest4(:,:,i);
        % From petro to elastic
        [Vp_temp,Vs_temp,Rho_temp]=applico_RPM_new([petro_rpm(:,3) petro_rpm(:,1) petro_rpm(:,2)]); 
    
        % Compute the new CMP with pred petro properties
        d_pre = (calcola_dati_pre_stack(Vp_temp,Vs_temp,Rho_temp,repmat(wavelet(:),1,length(ang)),ang));
        data_test4(:,:,i) = reshape(d_pre,[],length(ang));
        
        % Compute data error
        true_data = Xp_test(:,:,i);
        data_diff = data_test4(:,:,i)  - true_data ; % error
        data_MAE_test4(i) = mean(abs(data_diff(:)));
        data_PE_test4(i) = 100*(sum(abs(data_diff(:)))./sum(abs(true_data(:))));
        data_MSE_test4(i) = mean(data_diff(:).^2);
        data_RMSE_test4(i) = sqrt(data_MSE_test4(i));
  
    end
    
    sprintf('Data Percentage Error for test sample: %.2f', data_PE_test4(test_idx)) % Data error for chosen sample
    
    % Average data error
    disp(['Mean Data Error for phase shift of = ',num2str(phaseInDegrees(p)),'°'])
    sprintf('Avg Data Error: ')
    sprintf('Avg Data MAE:  %.3f', round(mean(data_MAE_test4,'omitnan'),4))   % Data mean squared error
    sprintf('Avg Data PE:   %.3f', round(mean(data_PE_test4, 'omitnan'),4))   % Mean over all test samples
    sprintf('Avg Data MSE:  %.3f', round(mean(data_MSE_test4, 'omitnan'),4))  % Mean over all test samples
    sprintf('Avg Data RMSE: %.3f', round(mean(data_RMSE_test4, 'omitnan'),4)) % Mean over all test samples

    % Compare two seismograms
    figure; 
    atldk(yax,ang,data_test4(:,:, test_idx),'color','r'); % pred
    xlabel('Angle (°)');ylabel('Time (s)');
    axis tight
    hold on
    atldk(yax,ang,Xp_test(:,:,test_idx),'color','k'); % true
    xlabel('Angle (°)');ylabel('Time');  
    title(['True vs Predicted Seismogram - ',num2str(phaseInDegrees(p)),'° phase shift'])
    axis tight       
    hold off
end
