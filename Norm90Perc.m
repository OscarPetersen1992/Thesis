function [ NormData, perc90 ] = Norm90Perc( data )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[r,c] = size(data);

vectorizedData = reshape(data,r*c,1);

perc90 = prctile(vectorizedData,90);

NormData = data/perc90;


end

