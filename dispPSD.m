function dispPSD( PSD, numSplits, F )
%Plotting the Power Spectral Density
%   PSD = PSD window
%   numSplits = number of time steps the PSD is divided in

LengthOneCase = length(PSD);
row = LengthOneCase/numSplits;
col=numSplits;

img = reshape(PSD,row,col);
% imagesc([1 3],F,10*log10(abs(img))) 
% %imagesc([1 24],F,10*log10(abs(spec))) 
% xlabel('Time (sec)')
% ylabel('Frequency (Hz)')
% % This sets the limits of the colorbar to manual for the first plot
% caxis manual
% caxis([-40 25]);
% c = colorbar;
% ylabel(c,'Power/frequency (dB/Hz)')
% set(gca,'Ydir','Normal')

plot(F,10*log10(img))
grid on
title('Periodogram')
xlabel('Frequency (Hz)')
ylabel('Power/Frequency (dB/Hz)')


end

