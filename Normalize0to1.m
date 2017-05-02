function normalized = Normalize0to1(array)

     [l,w] = size(array);
     % Normalize to [0, 1]
     normalized = zeros(l,w);    
     
     for i = 1:length(array)
         temp_data = array(i,:);
         m = min(temp_data);
         range = max(temp_data) - m;
         normalized(i,:) = (temp_data - m) / range;
     end
end

originalData = data.testData;

[r,c] = size(originalData);

oneRow = reshape(originalData,1,r*c);

normOneRow = zscore(oneRow');

normMatrix = reshape(normOneRow',r,c);


subplot(1,2,1)
dispPSD(originalData(1000,:),3)
subplot(1,2,2)
dispPSD(normMatrix(1000,:),3)
     