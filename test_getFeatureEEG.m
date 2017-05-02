%Test getFeature function for autoEncoder DBN in MNIST data set
clc
clear all;
more off;
addpath(genpath('DeepLearnToolboxGPU'));
addpath('DeeBNet');
data = EEG.prepareEEG('+EEG\',500000,50000);
data.normalize('meanvar');
%data.normalize('minmax');

%%
close all;
dbn=DBN();
dbn.dbnType='autoEncoder';
% Turn plot function off in GenerativeRBM.m - line 111

numEpochs=10;

tic

%dbn.dbnType='autoEncoder';
% RBM1
numHid = 250;
rbmParams=RbmParameters(numHid,ValueType.binary);
rbmParams.maxEpoch=numEpochs;
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.CD;
rbmParams.performanceMethod='reconstructionScale';
rbmParams.learningRate = 0.04;
rbmParams.hidBias=ones(1,numHid)*(-4);
dbn.addRBM(rbmParams);
% RBM2
% rbmParams=RbmParameters(100,ValueType.binary);
% rbmParams.maxEpoch=numEpochs;
% rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.CD;
% rbmParams.performanceMethod='reconstructionScale';
% dbn.addRBM(rbmParams);
% % RBM3
% rbmParams=RbmParameters(50,ValueType.binary);
% rbmParams.maxEpoch=numEpochs;
% rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.CD;
% rbmParams.performanceMethod='reconstructionScale';
% dbn.addRBM(rbmParams);  

dbn.train(data);
dbn.plotBases(1)
% dbn.backpropagation(data);
toc

%% plot
figure;
plotFig={'go' 'r+'};
for i=0:1
    img=data.testData(data.testLabels(:,1)==i,:);
    ext=dbn.getFeature(img);
    plot3(ext(:,1),ext(:,6),ext(:,9),plotFig{i+1});hold on;
end
legend('0','1');
hold off;
%% Reconstruct data

noisyData=data.testData(1:50,:);%sqrt(0.02).*randn(size(data.testData(1:9,:)));
[reconstructedData]=dbn.reconstructData(noisyData,5);

% Backtransform from log10 (-1 since transformation is log(1+spec))
noisyData = 10.^(noisyData)-1;
reconstructedData = 10.^(reconstructedData)-1;

%% Plot PSD

idx = 1;

testsets = [1:3 17 19 20];
close all 
 for ii=testsets
       
  subplot(2,6,idx)
  dispPSD(noisyData(ii,:),1);
  subplot(2,6,idx+6)
  dispPSD(reconstructedData(ii,:),1);
  
  idx = idx+1;
 
 end


%% Plot weights

index = 1:25;
numSplits = 1;
            
plotData = dbn.rbms{1,1}.rbmParams.weight(:,index);
LengthOneCase = length(plotData);
row = LengthOneCase/numSplits;
            
% Create colorbar for all subplots to retain comparability
mini = min(min(plotData));
maxi = max(max(plotData));

for i = 1:25
    img = reshape(plotData(:,i),row,numSplits);
    subplot(5,5,i)
    imagesc(img)
    set(gca,'YDir','normal')
    % This sets the limits of the colorbar to manual for the first plot
    caxis manual
    caxis([mini maxi]);
    colorbar;
end
 
 %% Plot EEG

idx = 1;

close all 
 for ii=4:6
       
  subplot(2,3,idx)
  plotEEG(data.testData(ii,:),3)
  subplot(2,3,idx+3)
  plotEEG(reconstructedData(ii,:),3)
  
  idx = idx+1;
 
 end
%%
 
wake=features_wLabels(features_wLabels(:,1)==0,:);
sleep=features_wLabels(features_wLabels(:,1)==1,:);

meanWake = mean(wake(:,2:end));
meanSleep = mean(sleep(:,2:end));

subplot(1,2,1)
W = dispPSD(meanWake,3);
subplot(1,2,2)
S = dispPSD(meanSleep,3);

% Combined
load('FreqAxes_0to45Hz');
CombinedIMG = [W S];
imagesc([1 3 6],F,10*log10(abs(CombinedIMG))) 
c = colorbar;
xlabel('Time (sec)')
ylabel('Frequency (Hz)')
ylabel(c,'Power/frequency (dB/Hz)')
set(gca,'Ydir','Normal')

%% 
 
subplot(2,3,1)
dispPSD(features_wLabels(5000,2:end),3)
subplot(2,3,2)
dispPSD(features_wLabels(17000,2:end),3)
subplot(2,3,3)
dispPSD(features_wLabels(23500,2:end),3)
subplot(2,3,4)
dispPSD(features_wLabels(10000,2:end),3)
subplot(2,3,5)
dispPSD(features_wLabels(11500,2:end),3)
subplot(2,3,6)
dispPSD(features_wLabels(13000,2:end),3)


 

   