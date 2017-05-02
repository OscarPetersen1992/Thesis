%% Main

path = '/Users/oscarpetersen/Documents/MATLAB/HypoSafe/ES042 - Pilot 2H/20151001_D_08/20151118';
% path = input('Please enter the path for the folder containing the data?'+...
%    '(Ex: C:\\Users\\X\\Desktop\\Bachelor): ','s');

patient = load('20151001_D_08.mat');
fs = patient.info.fs;
% Entering the right folder for reading in data.
if strcmp(path(1),'/') % Linux or Mac system
    fnames = dir([path,'/*00.mat']);
else % Windows system
    fnames = dir([path,'\*00.mat']);
end

number_of_channels = 2;

% Pre-allocating space for feature array
features_oneday = zeros(length(fnames),1+14);


for i = 1:length(fnames)
    
    fprintf('\n%d/%d \n',i,length(fnames))
    
    % Extracting file names
    fname = fnames(i).name;
    
    % Loading the extracted file name
    data = load(fname);
    eeg = data.eeg;
    
    % Number of features
    number_of_features = 4;
    
    % Pre-allocating space for feature array
    features_vec = zeros(1,number_of_features*number_of_channels);
    
    % Features for each channel
    for m = 1:number_of_channels
        
        % 1-4, wavelets.
        [cA, cD1] = dwt(eeg(:,m),'db5');
        [cA, cD2] = dwt(cA,'db5');
        [cA, cD3] = dwt(cA,'db5');
        [cA, cD4] = dwt(cA,'db5');
        [ ~, cD5] = dwt(cA,'db5');
        
        features_vec(3*m) = log(sum(abs(cD1)));
        features_vec(4*m) = log(sum(abs(cD2)));
        features_vec(5*m) = log(sum(abs(cD3)));
        features_vec(6*m) = log(sum(abs(cD4)));
        features_vec(7*m) = log(sum(abs(cD5)));
        
    end
    
    daytime = str2double(fname(1:2));
    features_oneday(i,1) = daytime;
    features_oneday(i,2:end) = features_vec;
    
end

%% Cluster analysis



%% K-means clustering

X = features_oneday([4:9 11 13 15] ,:);
% Number of clusters
K = 3;

% Run k-means
[i, Xc] = kmeans(X, K);

%% Plot results

% Plot data
clusterplot(X, y, i, Xc);

    
    