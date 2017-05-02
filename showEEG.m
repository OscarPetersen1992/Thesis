function [ output_args ] = showEEG(data, fs)
data = data';
% Defining the path to where the data is stored
% path = input(['Please enter the path for the folder containing the data?',...
%    '(Ex: C:\\Users\\X\\Desktop\\MATLAB): '],'s');

% data = load('/Users/oscarpetersen/Documents/MATLAB/HypoSafe/ES042 - Pilot 2H/20151001_D_08/20151013/1000.mat');
% data = data.eeg';

% Find the number of data points in each channel
num = length(data);

% Set the channels to be displayed
channels = 1:2;

% Calculate max and min in each row
mi = min(data,[],2);
ma = max(data,[],2);

% Calculates the absolute value between max and min in each channel. This
% is used to calculate the shift between each channel i.e. the position at
% the y-axis.
diffsum = cumsum([0; abs(ma(1:end-1))+abs(mi(2:end))]);

% Allocate space
h = zeros(1,length(diffsum)-1);

% Calculate the difference in shift between each channel
for i = 1:length(h)
    h(i) = diffsum(i+1)-diffsum(i);
end
mah = sort(h);


% Choosing a shift constant shift size between each channel
shift = linspace(0,mah(1)*length(channels),length(channels))';
shift = repmat(shift,1,num);

% Choose variable shift size between each channel
shift = repmat(diffsum,1,num);

t = linspace(0,num/fs,num);

%channelName = info.channel.name;
keyboard;
% plot 'eeg' data
plot(t,data+shift,'LineWidth',0.01)
axis equal
% edit axes
%set(gca,'ytick',mean(data+shift,2),'yticklabel',channelName)
grid on
ylim([min(1) max(max(shift+data))+1000])
xlim([0 max(t)])