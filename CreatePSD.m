function [F,T,X] = CreatePSD(tempdata,fs)

% This is outdated since the windowing takes place in get_feturesPSD

% Welch PSD and feature-extraction function
tempdata=tempdata';

[numCh, ~] = size(tempdata);
X = []; %Preallocate
window = fs * 3; % 3 sec window
nOverlap = fs * 2; % 1 sec overlap resulting in 3 timesteps in the 4 sec window;
nfft = fs;
numTimeSteps = 3;

% # of frequency bins in feature vector
% -3 to get below 50 Hz which is filtered
finalNumOfFreq = floor(nfft/4)-5;

% window = floor(numSamples/numTimeSteps);
% nOverlap = 0;

numCh = 1; % One channel

for ii = 1:numCh
    % each trial divided into 3 time-steps and the PSD is computed 
    % for each step
    
    % Using Hamming window
    % 
    %spectrogram(tempdata(ii,:),window,nOverlap,nfft,fs,'yaxis');
   [~,F,T,P] = spectrogram(tempdata(ii,:),window,nOverlap,nfft,fs);
   
   X =[X,P(1:finalNumOfFreq,:)];
    
end

F = F(1:finalNumOfFreq);


X = reshape(X,1,finalNumOfFreq*numTimeSteps*numCh);

check = max(X);

if check > 5000
    save('InsanePowerWindow.mat','tempdata')
    disp('Crazy')
end

%imagesc(10*log10(abs(P)))
% imagesc(T,F,10*log10(abs(P(1:finalNumOfFreq,:))))
% set(gca,'Xdir','Normal')
% set(gca,'Ydir','Normal')
% c = colorbar; 
% xlabel('Time (sec)')
% ylabel('Frequency (Hz)')
% ylabel(c,'Power/frequency (dB/Hz)')

end