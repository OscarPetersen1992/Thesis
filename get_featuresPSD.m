function [PSD_allWindows,F,T] = get_featuresPSD( data , fs)

% X = Normalized features
% y = Associated labels

% %% ---=== Windowing ===---
time = data(:,1:5);
label = data(:,6);
data = data(:,7:end);

[row,col] = size(data);

% 3 seconds of EEG
window_size = fs*3;

% Overlap 2 sec
step_size = fs*1;

% Pre-allocating space
window = zeros(window_size, col);

% Size of seizure array (+1 for the first window, which is substracted)
num_windows = (floor((row-window_size)/(step_size)))+1; % Number of windows

% Pre-allocating space for feature array
PSD_vec = [];
F = [];
T = [];

% Variable array for gathering the features from each window
PSD_allWindows = [];
RejectedWindows = [];

k = window_size; % First step size

% If window is larger than dataset
if k > row
    k = row;
end


while k <= row
    
    % Creating window
    window = data(k-window_size+1:k,:);
    
    % Check if window contains amplitudes above 150 muA
    check = sum([any(window>150) any(window<-150)]);
    
    if check==0
        
        [F,T,PSD_vec] = CreatePSD1(window,fs);
        
        % Label window
        check = sum(label(k-window_size+1:k,1));
        
        if check > window_size/4
            PSD_vec = [1 PSD_vec];
            
        else
            PSD_vec = [0 PSD_vec];
        end
        
        PSD_vec = [time(k,:) PSD_vec];
        % Idx 1:5 time (yyyy-mm-dd-hour-sec), minutes are left out
        % Idx 6 label
        % Idx 7:8 EEG
        
        
        %    % windowing EEG data instead
        %     EEG = window(:,1)';
        %     PSD_vec = [PSD_vec(1) EEG];
        
        % Array of all features for each window
        PSD_allWindows = [PSD_allWindows ; PSD_vec];
        
    else
        
        RejectedWindows = [RejectedWindows window];
        
    end
    
    if k > row-step_size && k < row % To get the last set of data point analyzed
        break
%         window_size = row-k;
%         k = row;
%         if window_size < 20 % 20 samples
%             k = row + 1;
%             num_windows = num_windows-1;
%         end
    else
        k = k + step_size; 
    end
    
     
    
    
end

% Save Rejected windows
%save('RejectedWindows.mat','RejectedWindows')

%features_allWindows(:,2:end) = zscore(features_allWindows(:,2:end));
%features_allWindows = zscore(features_allWindows);

% Alternative way of normalizing (divided by the 90th percentile)


%percentile90 = prctile(prctile(PSD_allWindows(:,2:end),90),90);
%PSD_allWindows(:,2:end) = PSD_allWindows(:,2:end)/percentile90;
 
end
