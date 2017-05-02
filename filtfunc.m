function Sout = filtfunc(s,lowpass,highpass)

Temp = filter(lowpass,s);


Sout = filtfilt(highpass,Temp);
end
