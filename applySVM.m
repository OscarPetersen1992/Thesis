function [ y_test_est, svmS ] = applySVM( X , y , time)
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
    time_train = time(CV.training(ii),:);
    X_test = X(CV.test(ii), :);
    y_test = y(CV.test(ii));
    time_test = time(CV.test(ii),:);
  
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
end

