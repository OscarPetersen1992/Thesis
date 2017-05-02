%% Sleep pattern plot

EEG_wLabels = matfile('EEG.mat');

fs = 207;

windowSize = 60*fs; % 1 min

numWindowsOneDay = (60*60*24*fs)/windowSize;

dayIdx = 1;
winCount = 1;
bpWindows = [];
labelWindows = [];

for i = 1:windowSize:length(EEG_wLabels)
    
    window_temp = EEG_wLabels(i:i+windowSize-1,:);
    
    bp = bandpower(window_temp(:,2), fs , [20 30]);
%     [cA, cD1] = dwt(window_temp,'db4');
%     [cA, cD2] = dwt(cA,'db4');
%     
%     features_temp(2) = log(sum(abs(cD1)));
%     bp = log(sum(abs(cD2)));
    
    if length(unique(window_temp(:,1))) == 2 
    	label_change = 1; 
    else
        label_change = 0;
    end
    
    bpWindows(winCount,dayIdx) = bp;
    labelWindows(winCount,dayIdx) = label_change;
    
    if winCount == numWindowsOneDay
        dayIdx = dayIdx + 1;
        winCount = 0;
    end
    
    winCount = winCount + 1;   
    
    
end

%% Removal of outliers

X = bpWindows;

percntiles = max(prctile(X, 90)); % 95th percentile
outlierIndex = X > percntiles;
%remove outlier values
X(outlierIndex) = percntiles;


%%
x=linspace(1,length(X(1,:))); 
y=linspace(0,24,6); 
colormap('jet')
imagesc(x,y,X);
colorbar;



    

