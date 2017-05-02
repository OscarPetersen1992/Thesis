function features_allWindows = get_features( data , fs)

% X = Normalized features
% y = Associated labels

% %% ---=== Windowing ===---
label = data(:,1);
data = data(:,2:end);

[row,col] = size(data);

% 30 seconds of EEG
window_size = fs*30;

% Overlap
window_size_overlap = floor(window_size*1);

% Pre-allocating space
window = zeros(window_size, col);

% Size of seizure array (+1 for the first window, which is substracted)
num_windows = (ceil((row-window_size)/(window_size_overlap)))+1; % Number of windows

% Number of features
number_of_features = 5;

% Pre-allocating space for feature array
features_temp = zeros(1, number_of_features);

% Variable array for gathering the features from each window
features_allWindows = [];
% http://se.mathworks.com/help/stats/pdf.html#bulrrtu-2

window_data = [];

% Defining and setting up the zero crossing detector
% Objectzerocross = dsp.ZeroCrossingDetector;

k = window_size; % First step size

% If window is larger than dataset
if k > row
    k = row;
end


while k <= row
    
    % Creating window
    window = data(k-window_size+1:k,:);
    
    features_vec = [];
    
    % Features for each channel
    for m = 1:col
        
        %         % 1.Mean
        %         features_temp(1) = mean(window(:,m));
        %
        %         % 2.Var
                  features_temp(1) = var(window(:,m));
        %
        %         % 3. Max-Min
        %         features_temp(3) = abs(max(window(:,m))-min(window(:,m)));
        %
%                   % 4.Zero-crossing
%                   % Count zero crossing in input
%                   features_temp(2) = step(Objectzerocross,window(:,m));
%         
%                   % 6. Bandpower of delta-band (0.5 - 4 Hz)
%                   features_temp(3) = bandpower(window(:,m),fs,[0.5 4]);
%                   % mean(spec(delta).*conj(spec(delta)));
%                   
%                   % 6. Bandpower of theta-band (4 - 8 Hz)
%                   features_temp(4) = bandpower(window(:,m),fs,[4 8]);
%                   
%                   % 6. Bandpower of theta-band (8 - 12 Hz)
%                   features_temp(5) = bandpower(window(:,m),fs,[8 12]);
        
        [cA, cD1] = dwt(window(:,m),'db4');
        [cA, cD2] = dwt(cA,'db4');
        [cA, cD3] = dwt(cA,'db4');
        [cA, cD4] = dwt(cA,'db4');
        [ ~, cD5] = dwt(cA,'db4');
        
        features_temp(2) = log(sum(abs(cD1)));
        features_temp(3) = log(sum(abs(cD2)));
        features_temp(4) = log(sum(abs(cD3)));
        features_temp(5) = log(sum(abs(cD4)));
        features_temp(6) = log(sum(abs(cD4)));
               
        features_vec = [features_vec features_temp];
        
    end
    
    % Label window
    check = sum(label(k-window_size+1:k,1));
     
    if check > window_size/4
        features_vec = [1 features_vec];
         
    else
        features_vec = [0 features_vec];
    end
        
    % Array of all features for each window 
    features_allWindows = [features_allWindows ; features_vec];
    
    
    if k > row-window_size_overlap && k < row % To get the last set of data point analyzed
        window_size = row-k;
        k = row;
%        release(Objectzerocross); % Released because of change of data size
        if window_size < 20 % 20 samples
            k = row + 1;
            num_windows = num_windows-1;
        end
    else
        k = k + window_size_overlap; % Overlap
    end
    
    
end

%features_allWindows(:,2:end) = zscore(features_allWindows(:,2:end));
%features_allWindows = zscore(features_allWindows);

end
