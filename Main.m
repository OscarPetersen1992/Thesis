%% UNEEG Medical - Master thesis
clear all 
clc

patient = 9;
pt = num2str(patient, '%02i');
date_start = 1;
date_end = 10;   

[features_wLabels,fs,F,T] = loadEEG(patient, date_start, date_end);
%[features_wLabels,EEG_wLabels,fs] = loadEEG(patient, date_start, date_end);

save('featuresFilt_wLabelsTime.mat','features_wLabels','F')

clear date_start date_end

%% Filtering raw EEG

load('filterCoeff.mat');
FilteredEEG = test_change;

for i = 1:5
     FilteredEEG(:,i) = filtfunc(FilteredEEG(:,i),lpfilt,hpfilt);
end

EEGPlot = FilteredEEG(:,3);

%% Plot EEG

fs = 207;
EEGPlot = tempdata(1,:);
num = length(EEGPlot);
t = linspace(0,num/fs,num);
plot(t,EEGPlot)
xlabel('Time (sec)')
ylabel('Amplitude (\muV)')
%axis([0 t(end) -30 30])


%% Plot PSD

PSDPlot = features_wLabels(1000,2:end);
numSplits = 3;
LengthOneCase = length(PSDPlot);
row = LengthOneCase/numSplits;
col=numSplits;

img = reshape(PSDPlot,row,col);
imagesc(10*log10(img)) 
c = colorbar;
xlabel('Time (sec)')
ylabel('Frequency (Hz)')
ylabel(c,'Power/frequency (dB/Hz)')

%% Removing Empty Rows

EEG_X_wLabels = EEG_wLabels;

% Removing empty rows
EEG_X_wLabels( ~any(EEG_X_wLabels(:,2:end),2), : ) = [];  %rows

%% Post processing

delay = 2; % sec

label_init = 0;

new_labels = zeros(length(y_test)-delay,1);
for i=delay+1:length(y_test)-delay % delay of 2 sec
    
    check = y_test_est(i-1)+y_test_est(i-2);%+y_test_est(i-3)+y_test_est(i-4);
    if check == delay
        label_init = 1;
    elseif check == 0
        label_init = 0;
    end
    
    new_labels(i) = label_init;
    
%     if check == delay
%         new_labels(i-delay) = 1;
%     elseif check == 0
%         new_labels(i-delay) = 0;
%     else
%         new_labels(i-delay) = y_test_est(i);
%     end
end

new_labels = [zeros(delay,1); new_labels];

measures = classperf(y_test,new_labels);

%% Plot of Learning rates

close all;
% HU BIAS -6
x = [0.005 0.01 0.05 0.1]; % Learning rates
y1 = [(0.1930+0.1957)/2 (0.1430+0.1764)/2 (0.1050+0.0998)/2 ...
    (0.0962+0.0962)/2]; % Performance

% HU BIAS -4
y2 = [(0.1269+0.1247)/2 (0.1113+0.1106)/2 (0.0989+0.0993)/2 ... 
    (0.0973+0.0970)/2]; % Performance

% HU BIAS -2
y3 = [(0.1140+0.1130)/2 (0.1093+0.1094)/2 (0.0973+0.0971)/2 ...
    (0.0953+0.0953)/2]; % Performance

% HU BIAS 0
y4 = [(0.1137+0.1138)/2 (0.1093+0.1101)/2  (0.0972+0.0970)/2 nan]; % Performance

figure; hold on
p1 = plot(x,y1); L1 = '-6';
p2 = plot(x,y2); L2 = '-4';
p3 = plot(x,y3); L3 = '-2';
p4 = plot(x,y4); L4 = '0';
legend([p1; p2; p3; p4], L1, L2, L3, L4);
xlabel('Learning Rate')
ylabel('Reconstruction error')
title('Optimization of HU Bias')


%% Amount of data - learning curve

x = [100000 250000 500000 750000 1000000 1500000 2000000];
y_rec_train_rbm1 = [0.1057 0.1043 0.1045 0.1041 0.1045 0.1037 0.1021];
y_rec_test_rbm1= [0.1047 0.1073 0.1093 0.1078 0.1041 0.1035 0.1013];

y_rec_train_rbm2 = [0.2631 0.2461 0.2515 0.2584 0.2394 0.2351 0.2475];
y_rec_test_rbm2= [0.2612 0.2454 0.2514 0.2565 0.24392 0.2345 0.2470];

plot(x,y_rec_train_rbm2)
hold on
plot(x,y_rec_test_rbm2)
title('RBM 2')
xlabel('Training samples')
ylabel('Reconstruction error')
legend('Train','Test');
hold off

%% Plot of misclassifications
data = new_labels;
xticklabels = [9:23 0:17];
xticks = linspace(1, size(data', 2), numel(xticklabels));

yticklabels = [0 1];
yticks = [0];

imagesc(data')
colormap('parula')
title('Classification overview')
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
 


%% Plot average spec, sleep vs awake
 testData= data.trainData;
 %[testData90,perc90]=Norm90Perc(testData);

specData = data.trainData;
labels = data.trainLabels;

awakeSpec = specData(labels == 0,:);
sleepSpec = specData(labels == 1,:);

meanAwakeSpec = mean(awakeSpec);
meanSleepSpec = mean(sleepSpec);

subplot(1,2,1)
dispPSD(meanAwakeSpec,1)
title('Awake')
subplot(1,2,2)
dispPSD(meanSleepSpec,1)
title('Sleep')

%%
features = get_features(EEG_X_wLabels, fs);

X = zscore(features(:,2:end));
y = features(:,1);

%save('featuresPt9.mat','X','y')

clear features EEG_X_wLabels 

%% PSD Analysis

PSD_allWindows = get_featuresPSD(EEG_X_wLabels,fs);

%save('PSD_allWindows','PSD_allWindows')

%% K-means clustering

% Number of clusters
K = 2;

[classNames, Xc] = kmeans(X(:,2:4), K,'MaxIter',10000);

% % Parralel pool
% pool = parpool;                      % Invokes workers
% stream = RandStream('mlfg6331_64');  % Random number stream
% options = statset('UseParallel',1,'UseSubstreams',1,...
%     'Streams',stream);
% 
% % Run k-means
% [i, Xc, sumd, D] = kmeans(X, K,'Options',options,'MaxIter',10000,...
%     'Display','final','Replicates',10);


%% 3D Scatter plot

class = y+1;
%class = classNames;

plot3(X(class==1,2),X(class==1,3),X(class==1,4),'r.')
hold on
plot3(X(class==2,2),X(class==2,3),X(class==2,4),'b.')
plot3(Xc(:,1),Xc(:,2),Xc(:,3),'kx','MarkerSize',15,'LineWidth',3)
legend('Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids'
grid on
hold off

%% K-means on sleep data only

X_sleep = X(y==1,:);

K = 3;

[class_sleep, Xc_sleep] = kmeans(X_sleep(:,2:4), K,'MaxIter',10000);

plot3(X_sleep(class_sleep==1,2),X_sleep(class_sleep==1,3),X_sleep(class_sleep==1,4),'r.')
hold on
plot3(X_sleep(class_sleep==2,2),X_sleep(class_sleep==2,3),X_sleep(class_sleep==2,4),'b.')
hold on
plot3(X_sleep(class_sleep==3,2),X_sleep(class_sleep==3,3),X_sleep(class_sleep==3,4),'g.')
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids'
grid on
hold off




%% Feature plot of the chosen channels


featNames = {'Variance','wave1','wave2','wave3','wave4'};

featData = X(:,1:5);
featLabels = y;


figure
gplotmatrix(featData,[],featLabels,'br',...
    'o',[],'on','',featNames,featNames)
title(sprintf('Patient: %s',pt,'Fontsize',14));
h = findobj('Tag','legend');
set(h, 'String', {'Wake', 'Sleep'})


clear featData featLabels featNames ii cha


%% PCA ANALYSIS
    
    [~, ~, latent] = pca(X);
    
    latent=latent/sum(latent);
    
    bar(latent)
  %  ygrid = linspace(0,5,5)*latent;
   % line(1:5,cumsum(latent));
    title(sprintf('Variance explained by principal components %s',pt));
    xlabel('Principal component');
    ylabel('Variance explained value');
    legend('Principal components','Cumulative explanation',2)
    ylim([0 1])
    grid on
    

%% Classification SVM

X = X(:,[2:4 7:9]);

K = 10;
CV = cvpartition(y,'KFold',K);

% Preallocating Variable for classification error
ErrorRate = zeros(K,1);
sensitivity_test = zeros(K,1);
specificity_test = zeros(K,1);

options.MaxIter = 10000000;
for ii = 1:K
    
fprintf('Crossvalidation fold %d/%d\n', ii, CV.NumTestSets);

    % Extract training and test set
    X_train = X(CV.training(ii), :);
    y_train = y(CV.training(ii));
    X_test = X(CV.test(ii), :);
    y_test = y(CV.test(ii));
  
    SVMStruct = svmtrain(X_train,y_train,'Options',...
        options,'BoxConstraint',0.01);
    y_test_est = svmclassify(SVMStruct, X_test);
 
    measures = classperf(y_test,y_test_est);
    sensitivity_test(ii) = measures.sensitivity;
    specificity_test(ii) = measures.specificity;
    ErrorRate(ii) = measures.errorrate;   

end

fprintf('Sensitivity: %.2f \n',mean(sensitivity_test))
fprintf('Specificity: %.2f \n',mean(specificity_test))
fprintf('Error rate: %.2f \n',mean(ErrorRate))

%% Deep Belief Network - Restricted Boltzmann Machines

addpath(genpath('/Volumes/HypoSafe/Thesis/DeepNeuralNetwork'))

X_DBN = X(1:10000, [2:4 7:9]);
y_DBN = y(1:10000);

K = 2;
CV = cvpartition(y,'KFold',K);

% Preallocating Variable for classification error
ErrorRate = zeros(K,1);
sensitivity_test = zeros(K,1);
specificity_test = zeros(K,1);

nodes = [6 10 10 2]; % [RBM(input) RBM RBM RBM(output)]
%bbdbn = randDBN( nodes, 'BBDBN' );
gbdbn = randDBN( nodes, 'GBDBN' );
bbdbn = gbdbn;
nrbm = numel(bbdbn.rbm); % Number of RBMs

opts.MaxIter = 1000;
opts.BatchSize = 100;
opts.Verbose = false; % Mute output from functions
opts.StepRatio = 0.1;
opts.object = 'CrossEntropy';

for ii = 1:K
    
    fprintf('Crossvalidation fold %d/%d\n', ii, CV.NumTestSets);
    
    % Extract training and test set
    X_train = X_DBN(CV.training(ii), :);
    y_train = y_DBN(CV.training(ii));
    X_test = X_DBN(CV.test(ii), :);
    y_test = y_DBN(CV.test(ii));
    
    opts.Layer = nrbm-1;
    bbdbn = pretrainDBN(bbdbn, X_train, opts);
    bbdbn= SetLinearMapping(bbdbn, X_train, y_train);
    
    disp('Step 1 completed')
    
    opts.Layer = 0;
    bbdbn = trainDBN(bbdbn, X_train, y_train, opts);
    
    disp('Step 2 completed')
    
    rmse= CalcRmse(bbdbn, X_train, y_train);
    ErrorRate= CalcErrorRate(bbdbn, X_train, y_train);
    fprintf( 'For training data:\n' );
    fprintf( 'rmse: %g\n', rmse );
    fprintf( 'ErrorRate: %g\n', ErrorRate );
    
    rmse= CalcRmse(bbdbn, X_test, y_test);
    ErrorRate= CalcErrorRate(bbdbn, X_test, y_test);
    fprintf( 'For test data:\n' );
    fprintf( 'rmse: %g\n', rmse );
    fprintf( 'ErrorRate: %g\n', ErrorRate );
    
end
