function [elas_fit, gof] = compute_Rsquared(x,y,label)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x and y have to be the same size  column vectors eg: 101 x 1
% label - is a string to label the x-axis/ specify the paramater being
% plotted
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[elas_fit,gof] = fit(double(x),double(y),'poly1');
figure;
plot(x',y','bo','MarkerFaceColor','b')
hold on
plot(elas_fit)
hold off
xlabel(['True ', label])
ylabel(['Predicted ',label])
legend('True vs Pred', 'Fitted Line','Location','southeast')
title(['Regression fit, R^2 = ',num2str(gof.rsquare)])
grid on
end 