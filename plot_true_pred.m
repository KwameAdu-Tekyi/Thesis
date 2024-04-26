function plot_true_pred(Yp_test,Yp_pred,yax,idx,test_number)
%%%
% Plot Sample of Obs vs Predicted parameters
% True and Pred are cell arrays 
% yax - y axis (correlation axis)
% idx - index of data from cell array to display 
% test number - test performed
% Also plots R2 fit to obs vs pred data 
%%%

true = double(Yp_test(:,:,idx));
pred = double(Yp_pred(:,:,idx));

% true=true(:,i);
% pred=pred(:,i);

head = sprintf('Test %d',test_number);
pet_labels = {'Phi', 'Sh', 'Sw'};
line_color = {'b', 'g', 'm'};

% Plot Petrophysical Parameters
figure('Name',head)
for i = 1 : size(true,2)
    subplot(1,3,i)
    plot(true(:,i),yax,line_color{i},'LineWidth',1.5);
    axis ij
    hold on
    plot(pred(:,i),yax,'--k','LineWidth',1.5); 
    axis ij
    ylabel('Time (s)')
    grid on
    title(pet_labels{i});
    
end
hold off

% Compute and plot R2
for i = 1:size(true,2) 
    % fit line to scatter plot of true vs obs data
    [elas_fit,gof] = fit(true(:,i),pred(:,i),'poly1');
    figure;
    plot(true(:,i),pred(:,i),'bo','MarkerFaceColor','b')
    hold on
    plot(elas_fit)
    hold off
    xlabel(['True ', pet_labels{i}])
    ylabel(['Predicted ',pet_labels{i}])
    legend('True vs Pred', 'Fitted Line','Location','southeast')
    title(['Regression fit, R^2 = ',num2str(gof.rsquare)])
    grid on
end

end