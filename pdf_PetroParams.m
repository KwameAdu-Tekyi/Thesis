function pdf_PetroParams(Facies,Petro,test_idx, petro_idx)
%%%%
% Plot the distribution of petrophysical parameters in each respective facies 
%%%%

label = {'SH','LPS','HPS'}; % Labels for petrophysical parameters
figure
for i = 1: size(Petro(:,:,test_idx),2)
    f1_in = double(Facies(:,:,test_idx))== i; % indices corresponding to Facies eg :Shale= 1
    Phi_f1 = Petro(f1_in,petro_idx,test_idx);
    %histogram(Phi_f1, BinWidth = 0.5, Normalization='pdf')
    Phi_f1_vals = linspace(-0.1,1.1 ,1000); %0 and 1 ideally
    pdf_f1 = fitdist(Phi_f1, "Normal");
    F = (1/pdf_f1.mu*sqrt(2*pi))*exp(-0.5*((Phi_f1_vals - pdf_f1.mu)./pdf_f1.sigma).^2); 
    F_norm = F./max(F);
    plot(Phi_f1_vals,F_norm,'LineWidth',2)
    hold on 
    grid on
    xlim([min(Phi_f1) max(Phi_f1)])
end
    hold off
    legend(label, 'location', 'best')
    ylabel('Probability')
    axis tight 


end

