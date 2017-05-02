function dispPSD6( features , numSplits)

features = features';

[LengthOneCase,numImages] = size(features);
row = LengthOneCase/numSplits;

col=numSplits;

img2=zeros(2*row,col*3);

for nn=1:numImages
  ii=rem(nn,2); 
  if(ii==0) 
      ii=2; 
  end
  jj=ceil(nn/2);
  img1 = reshape(10*log10(abs(features(:,nn))),col,row);
  img2(((ii-1)*row+1):(ii*row),((jj-1)*col+1):(jj*col))=img1';
end

% X = img2;
% 
% percntiles = max(prctile(X, 95)); % 95th percentile
% outlierIndex = X > percntiles;
% %remove outlier values
% X(outlierIndex) = percntiles;

imagesc(img2)
colorbar;
drawnow;
err=0; 
end

