function dispPSD( PSD, numSplits, F )
%Plotting the Power Spectral Density
%   PSD = PSD window
%   numSplits = number of time steps the PSD is divided in

LengthOneCase = length(PSD);
row = LengthOneCase/numSplits;
col=numSplits;

load('FreqAxes_0to45Hz');

img = reshape(PSD,row,col);
imagesc([1 3],F,10*log10(abs(img'))) 
c = colorbar;
xlabel('Time (sec)')
ylabel('Frequency (Hz)')
ylabel(c,'Power/frequency (dB/Hz)')
set(gca,'Ydir','Normal')


end

