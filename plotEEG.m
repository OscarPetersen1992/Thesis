function plotEEG( EEG , fs )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

num = length(EEG);
t = linspace(0,num/fs,num);
plot(t,EEG)
xlabel('Time (sec)')
ylabel('Amplitude (\muV)')

end

