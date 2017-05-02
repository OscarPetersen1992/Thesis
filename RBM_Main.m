% Modified by Oscar Petersen
%
% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton  
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our 
% web page. 
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.


% This program pretrains a deep autoencoder for MNIST dataset
% You can set the maximum number of epochs for pretraining each layer
% and you can set the architecture of the multilayer net.

clear all
close all

RBM_path = '/Volumes/HypoSafe/Thesis/RBM';

addpath(RBM_path);

PSD_wLabels = load('PSD_allWindows.mat');
PSD_wLabels = PSD_wLabels.PSD_allWindows;

PSD_wLabels(:,2:end) = zscore(PSD_wLabels(:,2:end));

%PSD_wLabels(:,2:end) = Normalize0to1(PSD_wLabels(:,2:end));

maxepoch=30; %In the Science paper we use maxepoch=50, but it works just fine. 
numhid=300; numpen=150; numpen2=75; %numopen=156;
 

fprintf(1,'Pretraining a deep autoencoder. \n');

[batchdata, batchtargets, testbatchdata, testbatchtargets] = makebatches(PSD_wLabels);

[numcases numdims numbatches]=size(batchdata);

fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
restart=1;
rbm;
hidrecbiases=hidbiases; 
save mnistvh vishid hidrecbiases visbiases;

fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);
batchdata=batchposhidprobs;
numhid=numpen;
restart=1;
rbm;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;
save mnisthp hidpen penrecbiases hidgenbiases;

fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d \n',numpen,numpen2);
batchdata=batchposhidprobs;
numhid=numpen2;
restart=1;
rbm;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
save mnisthp2 hidpen2 penrecbiases2 hidgenbiases2;

% fprintf(1,'\nPretraining Layer 4 with RBM: %d-%d \n',numpen2,numopen);
% batchdata=batchposhidprobs;
% numhid=numopen; 
% restart=1;
% rbmhidlinear;
% hidtop=vishid; toprecbiases=hidbiases; topgenbiases=visbiases;
% save mnistpo hidtop toprecbiases topgenbiases;

%backprop; 

load mnistvh
load mnisthp
load mnisthp2
load mnistpo 

[batchdata, batchtargets, testbatchdata, testbatchtargets] = makebatches(PSD_wLabels);

% %%%% PREINITIALIZE WEIGHTS OF THE AUTOENCODER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% w1=[vishid; hidrecbiases];
% w2=[hidpen; penrecbiases];
% w3=[hidpen2; penrecbiases2];
% w4=[hidtop; toprecbiases];
% w5=[hidtop'; topgenbiases]; 
% w6=[hidpen2'; hidgenbiases2]; 
% w7=[hidpen'; hidgenbiases]; 
% w8=[vishid'; visbiases];

%%%% PREINITIALIZE WEIGHTS OF THE AUTOENCODER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w1=[vishid; hidrecbiases];
w2=[hidpen; penrecbiases];
w3=[hidpen2; penrecbiases2];
w4=[hidpen2'; hidgenbiases2]; 
w5=[hidpen'; hidgenbiases]; 
w6=[vishid'; visbiases];

%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l1=size(w1,1)-1;
l2=size(w2,1)-1;
l3=size(w3,1)-1;
l4=size(w4,1)-1;
l5=size(w5,1)-1;
l6=size(w6,1)-1;
% l7=size(w7,1)-1;
% l8=size(w8,1)-1;
% l9=l1; 
test_err=[];
train_err=[];

[numcases numdims numbatches]=size(batchdata);


%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err=0; 
[numcases numdims numbatches]=size(batchdata);
N= numcases;

for batch = 1:numbatches

  data = [batchdata(:,:,batch)];
  data = [data ones(N,1)];
%   w1probs = 1./(1 + exp(-data*w1)); 
%   w1probs = [w1probs  ones(N,1)];
%   w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
%   w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
%   %w4probs = 1./(1 + exp(-w3probs*w4)); w4probs = [w4probs  ones(N,1)];
%   w4probs = w3probs*w4; w4probs = [w4probs  ones(N,1)];
%   w5probs = 1./(1 + exp(-w4probs*w5)); w5probs = [w5probs  ones(N,1)];
%   w6probs = 1./(1 + exp(-w5probs*w6)); w6probs = [w6probs  ones(N,1)];
%   w7probs = 1./(1 + exp(-w6probs*w7)); w7probs = [w7probs  ones(N,1)];
%   dataout = 1./(1 + exp(-w7probs*w8));
%   err= err +  1/N*sum(sum( (data(:,1:end-1)-dataout).^2 )); 
%   train_err(epoch)=err/numbatches;

  w1probs = 1./(1 + exp(-data*w1)); 
  w1probs = [w1probs  ones(N,1)];
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
  w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
  w3probs = w2probs*w3; w3probs = [w3probs  ones(N,1)];
  w4probs = 1./(1 + exp(-w3probs*w4)); w4probs = [w4probs  ones(N,1)];
  %w4probs = w3probs*w4; w4probs = [w4probs  ones(N,1)];
  w5probs = 1./(1 + exp(-w4probs*w5)); w5probs = [w5probs  ones(N,1)];
  dataout = 1./(1 + exp(-w5probs*w6));
  err= err +  1/N*sum(sum( (data(:,1:end-1)-dataout).^2 )); 
  train_err(epoch)=err/numbatches;
  
  fprintf(1,'Displaying in figure 1: Top row - real data, Bottom row -- reconstructions \n');
    output=[];
 for ii=1:3
  output = [output data(ii,1:end-1)' dataout(ii,:)'];
 end
 
 figure('Position',[100,600,1000,200]);
 mnistdisp(output);
 drawnow;
 
 pause(1)
 close all
  
end


 


%%%%%%%%%%%%%% END OF COMPUTING TRAINING RECONSTRUCTION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


