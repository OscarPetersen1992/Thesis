%function [ features_total,concat_EEG, fs] = loadEEG( patient, date_start, date_end )
function [ features_total,fs,F,T ] = loadEEG( patient, date_start, date_end )

Master_path = '/Volumes/HypoSafe/Thesis/ES042 - Pilot 2H/';

addpath(genpath(Master_path));

% Directory of patients (Both day & night)
q = dir(Master_path);
q=q(~ismember({q.name},{'.','..','._Main.m'}));

patients = {q.name};
clear q;

% Current patient
pt = num2str(patient, '%02i');

% Creating path and search for EEG files for the given patient
current_patient_day = strcat('D_',pt);
ix=regexp(patients,current_patient_day);
if cellfun('isempty',ix)
    fprintf('Patient %s does not exist.\n', pt);
else
    ix=~cellfun('isempty',ix);
    current_patient_day=patients{ix};
    
    % Check if there exist a night file
    current_patient_night = strcat('N_',pt);
    ix=regexp(patients,current_patient_night);
    if cellfun('isempty',ix)
        fprintf('Patient %s has no night file.\n', pt);
    else
        ix=~cellfun('isempty',ix);
        current_patient_night=patients{ix};
    end
end


% Creates path for the given patient (Day & Night)
path_day =strcat(Master_path,current_patient_day);
dates= dir(path_day);
dates_day = {dates.name};

path_night = strcat(Master_path,current_patient_night);
dates = dir(path_night);
dates_night = {dates.name};

clear ix;

% Loading scored data if available
ix=regexp(dates_night,'EpiView_marks');
if cellfun('isempty',ix)
    fprintf('Patient %s has no scored EEG.\n', pt);
else
    ix=~cellfun('isempty',ix);
    EpiView = load(dates_night{find(ix)});
end

% Loading sleep & wake time stamps from Sirin
TimeStampFile = ['SleepWakeTimeStampsPt' num2str(patient) '.mat'];
if exist(TimeStampFile, 'file');
    % Named as sleepwake_datenums
    load(TimeStampFile);
else
    fprintf('Patient %s has no time stamps.\n', pt);
end


% Load fs
patient_info = load([path_day '/' current_patient_day '.mat']);
fs = patient_info.info.fs;


% Removal of all unused folders & files (fx. _LOG, _ACC files etc.)

% Day
contains_digits = isstrprop(dates_day, 'digit');
find_dates = cellfun(@sum,contains_digits);
idx = find(find_dates == 8);
dates_day= dates_day(idx);

% Night
contains_digits = isstrprop(dates_night, 'digit');
find_dates = cellfun(@sum,contains_digits);
idx = find(find_dates == 8);
dates_night=dates_night(idx);

clear contains_digits find_dates idx

%% Split of EpiView file

[~,intersecStampVSEpi,~]=intersect(EpiView.marks.time, sleepwake_datenums);

comments = EpiView.marks.comment(intersecStampVSEpi);
date_EpiView = datevec(datetime(EpiView.marks.time(intersecStampVSEpi),'ConvertFrom','datenum'));

% ix = find(~strcmp(comments,'Sleep start') & ~strcmp(comments,'Sleep end'));
% % Removal of anything but sleep & wake comments
% comments(ix,:) = [];
% date_EpiView(ix,:) = [];
% clear ix;


%% Concat day & night for each date

path_day1 = strcat(path_day,'/');
path_night1 = strcat(path_night,'/');

concat_EEG=[];
features_total = [];
features_temp = [];


sdate = datenum(str2double(dates_day{date_start}(1:4)),str2double(dates_day{date_start}(5:6)),str2double(dates_day{date_start}(7:8)),0,0,0);
edate = datenum(str2double(dates_day{date_end}(1:4)),str2double(dates_day{date_end}(5:6)),str2double(dates_day{date_end}(7:8))+1,0,0,0);
%edate = datenum(str2double(dates_day{end}(1:4)),str2double(dates_day{end}(5:6)),str2double(dates_day{end}(7:8)),0,0,0);
% 1 day is added to end date

%length of time step is 1 hour
delta=datenum(0,0,0,1,0,0);
idx_label = 1;
label = 0;

load('filterCoeff.mat');

time_sec = [];
for sec = 0:3599
    time_sec = [time_sec; ones(fs,1)*sec];
end

for nw=sdate:delta:edate
    
    datestr(nw);
    
    temp_hour = datevec(datetime(nw,'ConvertFrom','datenum'));
    
    date_str = [num2str(temp_hour(1),'%02i') num2str(temp_hour(2),'%02i') num2str(temp_hour(3), '%02i') '/'];
    
    path_day_date = [path_day1 date_str num2str(temp_hour(4),'%02i') '00.mat' ];
    path_night_date = [path_night1 date_str num2str(temp_hour(4),'%02i') '00.mat'];
    
    day_file_exist = exist(path_day_date, 'file');
    night_file_exist = exist(path_night_date, 'file')*2;
    
    check_if_exist = sum([day_file_exist night_file_exist]);
    
    if check_if_exist == 6
        %Both day & night EEG exist
        temp_eeg_day = load(path_day_date);
        temp_eeg_day = temp_eeg_day.eeg;
        
        temp_eeg_night = load(path_night_date);
        temp_eeg_night = temp_eeg_night.eeg;
        
        temp_eeg = temp_eeg_day + temp_eeg_night;
        
        
    elseif check_if_exist == 4
        % Only night EEG exists
        temp_eeg_night = load(path_night_date);
        temp_eeg = temp_eeg_night.eeg;
        
    elseif check_if_exist == 2
        % Only day EEG exists
        temp_eeg_day = load(path_day_date);
        temp_eeg = temp_eeg_day.eeg;
        
    else
        fprintf('No file for hour %s.\n', datestr(nw))
        %temp_eeg = [];
        temp_eeg = zeros(745200,2);
    end
    
    
    % If the loop date & time match a scored date & time from EpiView
    if isequal(temp_hour(1:4), date_EpiView(idx_label, 1:4));
        samples_b4_label_change = (date_EpiView(idx_label,5)*60+floor(date_EpiView(idx_label,5)))*fs;
        if strcmp(comments{idx_label},'Sleep start')
            before = zeros(samples_b4_label_change,1);
            after = ones(length(temp_eeg)-length(before),1);
            labels = [before ; after];
            label = 1;
        elseif strcmp(comments{idx_label},'Sleep end')
            before = ones(samples_b4_label_change,1);
            after = zeros(length(temp_eeg)-length(before),1);
            labels = [before ; after];
            label = 0;
        else fprintf('Labeling problem: %d\n', temp_hour)
        end
        idx_label = idx_label + 1;
    else
        labels = ones(length(temp_eeg),1)*label;
    end
    
    % Adding time label
    hour_sec = [repmat(temp_hour(1:4),length(labels),1) time_sec];
    temp_labeled_eeg = [hour_sec labels temp_eeg];
    
    % Calculating features for each file
    % Removing empty rows
    temp_labeled_eeg( ~any(temp_labeled_eeg(:,7:end),2), : ) = [];  %rows
    
    if ~isempty(temp_labeled_eeg)
        
        % Filter if array contains non-zero elements
        for i = 7:8
            temp_labeled_eeg(:,i) = filtfunc(temp_labeled_eeg(:,i),lpfilt,hpfilt);
        end
        
        % Removal of first 5 sec filtering due to stabilization
        temp_labeled_eeg(1:fs*5,:) = [];
        
        [features_temp,F,T] = get_featuresPSD(temp_labeled_eeg,fs);
        
        features_total = [features_total; features_temp];
       
        
    end
    
    %concat_EEG = [concat_EEG ; temp_labeled_eeg];
end


end

