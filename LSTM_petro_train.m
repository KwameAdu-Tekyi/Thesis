% Run main.m first
clc

X = Data;
YPetro = Petro;
rng(42);
l = size(Data,3);
n = 0.7; % training proportion
idx_split = n*l;
val_size = 0.15;

% Elastic Parameters for seismogram computation
Ye = Elas;
Ye_test = Ye(:,:,(idx_split+val_size*l)+1:end);

% Petrophsyical parameters (Phi,Sh,Sw)
XTrain = X(:,:,1:idx_split);
Yp_train = YPetro(:,:,1:idx_split);
Xp_val = X(:,:,idx_split + 1:idx_split+ val_size*l);
Yp_val = YPetro(:,:,idx_split + 1:idx_split+val_size*l);
Xp_test = X(:,:,(idx_split)+(val_size*l)+1 :end);
Yp_test = YPetro(:,:,(idx_split)+(val_size*l)+1 :end);

%test_idx = randi(length(Yp_test),1);
test_idx = 393; %614
figure;  
atldk(yax,ang,Xp_test(:,:,test_idx),'wiggle','accurate','color','k');
xlabel('Incidence Angle °')
axis tight
title('Sample Test Seismogram')

%% Building the LSTM network

numFeatures = size(XTrain,2); % No of angles
miniBatchSize = 24; %numel(Data)/10  32 24

%numHiddenUnits = 50; 
maxEpochs = 10; % 5
%rng(42);

% Regression (Petro x'tics)
numResponses = size(Yp_train,2); %Phi, Sh,Sw =  No of variables to predict 
layers = [
    sequenceInputLayer(numFeatures)
    batchNormalizationLayer
    bilstmLayer(100,'OutputMode','sequence') % 1
    %fullyConnectedLayer(20) % new added
    dropoutLayer(0.3)
    leakyReluLayer


    % batchNormalizationLayer
    % bilstmLayer(64) % 50
    % %fullyConnectedLayer(20) % new added
    % dropoutLayer(0.3)
    % batchNormalizationLayer
    % leakyReluLayer
    % 
    % batchNormalizationLayer
    % bilstmLayer(24) % 50
    % %fullyConnectedLayer(10) % new added
    % dropoutLayer(0.1)
    % batchNormalizationLayer
    leakyReluLayer

    fullyConnectedLayer(numResponses)];

%Convert the layers to a layerGraph
lgraph = layerGraph(layers);

% Loss Function
lossFcn = "mean-absolute-error";

% Convert the layer graph to a dlnetwork
dlnet = dlnetwork(lgraph);

% Training Options 
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.001, ... %0.001
    'GradientThreshold',1, ...
    'Shuffle','every-epoch', ...
    'Metrics','rmse', ...
    'Plots','training-progress',...
    'Verbose',0,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.2,... %0.2
    'ValidationData',{Xp_val,Yp_val},...
    'ValidationFrequency',50,...
    'ValidationPatience',10,...
    'L2Regularization',0.001); %bsf 0.001 0.01

% Train the network
%net_petro = trainnet(XTrain,Yp_train,dlnet,lossFcn, options);
%% Save best network as in .mat file called net_petro_.....mat
%save('net_petro_100.mat','net_petro');

%load/ recall network from disk

s = load('net_petro_100.mat'); 
net_petro = s.net_petro;
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

%% ----------------------------Test 2--------------------------------------
% Error in noise estimation---> 10% of the std of test data

rng(42);
% Generate new syn. data and add noise 
Xp_test2 = zeros(size(Xp_test));
var_level = 0.10; % 10 percent

for c = 1 : size(Xp_test,3)
    % Variance of XTest 
    desired_variance  = var_level .* var(Xp_test(:,:,c)); % 10 percent of the variance of test 1 data
    noise_Xp = sqrt(desired_variance).* randn(size(Xp_test(:,:,c))); % scaled noise
    % Add noise to test seismogram, Xp_test
    Xp_test2(:,:,c) =  Xp_test(:,:,c) + noise_Xp;

end

% Visualize noisy Seismogram
figure
atldk(yax,ang,Xp_test2(:,:,test_idx),'color','r'); % Noisy 
hold on 
atldk(yax,ang,Xp_test(:,:,test_idx),'color','k'); % true
hold off
title('Noisy vs Original Test Seismogram')
xlabel('Incidence Angle (°)')
axis tight

% Petrophysical LSTM predictions on noisy data
Yp_predTest2 = minibatchpredict(net_petro,Xp_test2,'MiniBatchSize',1);

% Evaluate Predictions
errors_NN(Yp_test, Yp_predTest2,test_idx,2);

% Visualize predictions : Predicted vs True Petrophysical properties
plot_true_pred(Yp_test,Yp_predTest2,yax,test_idx,2);

% Avg RMS and % Err for entire test set
rmse_NN_test2 =  zeros(size(Yp_test,3),3);    % RMSE
pe_NN_test2 = zeros(size(Yp_test,3),3);       % Perc Error
r2_NN_test2 = zeros(size(Yp_test,3),3);

for i = 1:size(Yp_test,3)
    err = Yp_predTest2(:,:,i) - Yp_test(:,:,i);
    rmse_NN_test2(i,:) = sqrt(mean(err.^2));
    for j = 1 : size(pe_NN_test2,2)
       diff = Yp_predTest2(:,j,i) - Yp_test(:,j,i);
       pe_NN_test2(i,j) = 100*(sum(abs(diff))./sum(abs(Yp_test(:,j,i))));
       xtemp = double(Yp_test(:,j,i));
       ytemp = double(Yp_predTest2(:,j,i));
       [~,gof_petro] = fit(xtemp,ytemp,'poly1');
       r2_NN_test2(i,j) = gof_petro.rsquare;
    end
end

rmse_NN_test2 = mean(rmse_NN_test2);        % Mean over all test samples
pe_NN_test2 = mean(pe_NN_test2);            % Mean over all test samples
r2_NN_test2 = mean(r2_NN_test2);            % Mean over all test samples

sprintf('Avg RMSE error for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
    rmse_NN_test2(1),rmse_NN_test2(2),rmse_NN_test2(3))
sprintf('Avg Percentage error for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
    pe_NN_test2(1),pe_NN_test2(2),pe_NN_test2(3))
sprintf('Avg R2 for all test samples\nPhi: %.2f \nSh: %.2f,\nSw: %.2f',...
    r2_NN_test2(1),r2_NN_test2(2),r2_NN_test2(3))

% Data Error for Test2
data_test2 = zeros(size(Xp_test));
data_MAE_test2 = zeros(size(Xp_test,3),1);
data_PE_test2 = zeros(size(Xp_test,3),1);
data_MSE_test2 = zeros(size(Xp_test,3),1);
data_RMSE_test2 = zeros(size(Xp_test,3),1);

% Compute seismograms from predicted petrophysical parameters
for i = 1: size(Yp_test,3)
    petro_rpm = Yp_predTest2(:,:,i);
    % From petro to elastic
    [Vp_temp,Vs_temp,Rho_temp]=applico_RPM_new([petro_rpm(:,3) petro_rpm(:,1) petro_rpm(:,2)]); 
    
    % Compute the new CMP with pred petro properties
    d_pre = (calcola_dati_pre_stack(Vp_temp,Vs_temp,Rho_temp,repmat(wavelet(:),1,length(ang)),ang));
    data_test2(:,:,i) = reshape(d_pre,[],length(ang));
    
    % Compute data error
    true_data = Xp_test(:,:,i);
    data_diff = data_test2(:,:,i)  - true_data ; % error/ difference
    data_MAE_test2(i) = mean(abs(data_diff(:)));
    data_PE_test2(i) = 100*(sum(abs(data_diff(:)))./sum(abs(true_data(:))));
    data_MSE_test2(i) = mean(data_diff(:).^2);
    data_RMSE_test2(i) = sqrt(data_MSE_test2(i));
end

sprintf('Data Percentage Error for test sample: %.2f', data_PE_test2(test_idx)) % Average data error

% Average data error
sprintf('Avg Data Error: ')
sprintf('Avg Data MAE:  %.3f', round(mean(data_MAE_test2,'omitnan'),4))   % Data mean squared error
sprintf('Avg Data PE:   %.3f', round(mean(data_PE_test2, 'omitnan'),4))   % Mean over all test samples
sprintf('Avg Data MSE:  %.3f', round(mean(data_MSE_test2, 'omitnan'),4))  % Mean over all test samples
sprintf('Avg Data RMSE: %.3f', round(mean(data_RMSE_test2, 'omitnan'),4)) % Mean over all test samples

% Compare two seismograms
figure; atldk(yax,ang,data_test2(:,:,test_idx),'color','r'); % pred
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
 
%% Test 4: Erroneous Peak Frequency and phase of source wavelet

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

%% Test 5:  Error In Assumed Prior Elastic and Facies Models

% a)Tweak the transition matrix to increase shale probability by 5%
Tmat_test5 = [
    0.95 0.025 0.025;...
    0.15 0.85 0;...
    0.15 0.05 0.8];
%Tmat_test5 = Tmat;

% b)Tweak covariance matrix to in increase errors to 5%
% Cm_facies_test5 = Cm_facies;
Cm_facies_test5 = 1.05 .* Cm_facies;

% Tweak mean
Mu_facies_test5 = 1.05.*Mu_facies;
%Mu_facies_test5= Mu_facies;
% c) Both a and b
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
    for j = 1 : size(pe_NN_test5,2)
        true_data = Yp_test(:,j,i);
        diff = Yp_predTest5(:,j,i)- true_data ;
        pe_NN_test5(i,j) = 100*(sum(abs(diff))./sum(abs(true_data)));
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

%% Test 6:  Error In Assumed Rock Model
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