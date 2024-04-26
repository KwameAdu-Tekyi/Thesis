function [PE,RPE,MSE,RMSE,MAE]= errors_NN(Yp_test, Yp_pred, idx,test_no)

% Variables to store computed errors
PE = zeros(size(Yp_test(:,:,idx),2),1);     % Percentage Error
RPE = zeros(size(Yp_test(:,:,idx),2),1);    % Relative Percentage Error
RMSE = zeros(size(Yp_test(:,:,idx),2),1);   % Root Mean Squared Error
MSE = zeros(size(Yp_test(:,:,idx),2),1);    % Mean Squared Error
MAE = zeros(size(Yp_test(:,:,idx),2),1);    % Mean Absolute Error
R2 =  zeros(size(Yp_test(:,:,idx),2),1);    % R2 - (R Squared)

fprintf('RESULTS FROM TEST %d \n',test_no)

% Error = measured - actual
for i = 1:3
    true = Yp_test(:,:,idx);
    true = true(:,i);
    pred = Yp_pred(:,:,idx);
    pred = pred(:,i);
    diff = pred(:) - true(:); % pred - obs
    
    if i==1
        %PHI
        disp('Phi Percentage Error:')
        PE(i) = 100*(sum(abs(diff))./sum(abs(true)));
        % 100*(sum(abs(err))./sum(abs(true))));
        fprintf('%.2f\n',PE(i))
                
        disp('Phi MSE:');
        MSE(i) = mean(diff.^2);
        fprintf('%.2f\n',MSE(i));
            
        disp('Phi RMSE:')
        % RMSE(i) = sqrt(mean(diff.^2));
        RMSE(i) = sqrt(MSE(i));
        fprintf('%.2f\n',RMSE(i))
           
        disp('Phi MAE:')
        MAE(i) = mean(abs(diff));
        fprintf('%.2f\n',MAE(i))
        
        disp('Phi R2: ')
        x = true;
        y = pred;
        [~,gof_phi] = fit(double(x),double(y),'poly1');
        R2(i) = gof_phi.rsquare;
        fprintf('%.2f\n',R2(i))
          
    elseif i==2
        %SH
        disp('Sh Percentage Error:')
        PE(i) = 100*(sum(abs(diff))./sum(abs(true(:))));
        fprintf('%.2f\n',PE(i))
                
        disp('Sh MSE:');
        MSE(i) = mean(diff.^2);
        fprintf('%.2f\n',MSE(i))
            
        disp('Sh RMSE:')
        RMSE(i) = sqrt(mean(diff.^2,"all"));
        %RMSE(i) = sqrt(MSE(i));
        fprintf('%.2f\n',RMSE(i))
         
        disp('Sh MAE:')
        MAE(i) = mean(abs(diff));
        fprintf('%.2f\n',MAE(i))
        
        disp('Sh R2: ')
        x = true;
        y = pred;
        [~,gof_phi] = fit(double(x),double(y),'poly1');
        R2(i) = gof_phi.rsquare;
        fprintf('%.2f\n',R2(i))  

    else
        %SW
        disp('Sw Percentage Error:')
        PE(i) = 100*(sum(abs(diff))./sum(abs(true(:))));
        fprintf('%.2f\n',PE(i))
                
        disp('Sw MSE:');
        MSE(i) = mean(diff.^2);
        fprintf('%.2f\n',MSE(i))
            
        disp('Sw RMSE:')
        RMSE(i) = sqrt(mean(diff.^2,"all"));
        %RMSE(i) = sqrt(MSE(i));
        fprintf('%.2f\n',RMSE(i))
            
        disp('Sw MAE:')
        MAE(i) = mean(abs(diff));
        fprintf('%.2f\n',MAE(i))
        
        disp('Sw R2: ')
        x = true;
        y = pred;
        [~,gof_phi] = fit(double(x),double(y),'poly1');
        R2(i) = gof_phi.rsquare;
        fprintf('%.2f\n',R2(i))
            
    end
end