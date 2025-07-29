clear;
clc;
close all;
load cifar-10-data.mat;

% random number maker
rng(38811103);
classes_hor = randperm(10, 3);

%reformat the data
for p=1:3
classes(p,1)=classes_hor(1,p);
end

extract = [];
nLabel = [];

% get the data needed to the new array
for c = 1:60000
    if ismember(labels(c), classes_hor)
        extract = [extract; c];
        nLabel = [nLabel; labels(c)];
    end
end

% Create new data arrays
nData = data(extract, :, :, :);

%Random generator for part 2
rng(38811103)
rx = randperm(18000,18000);

%reformat and loading the Training index
for px=1:9000
training_index(px,1)=rx(1,px);
end

%reformat and loading the Testing index
for px=1:9000
testing_index(px,1)=rx(1,9000+px);
end

% load training data and label
trainingData= nData(training_index,:,:,:);
trainingLabel=nLabel(training_index,1);

%load test data and label
testingData=nData(testing_index,:,:,:);
testingLabel=nLabel(testing_index,1);

% Flatten the training datas
flatTrainingData = single(reshape(trainingData, 9000, 3072));
flatTestingData = single(reshape(testingData, 9000, 3072));

%setting up number of neighbours
k = 5;

% Trying to match with Euclidean distance
tic;
dist_euc = sqrt(sum((flatTestingData.^2), 2) + sum((flatTrainingData.^2), 2)' - 2 * (flatTestingData * flatTrainingData'));
[~, sorted_euc] = sort(dist_euc, 2);
nearest_euc = trainingLabel(sorted_euc(:, 1:k));
predict_euc = mode(nearest_euc, 2);
knnL2_timetaken = toc;
fprintf("EUC Done\n");

% Trying to match by Cosine distance
tic;
sqrt_flatTestingData = sqrt(sum(flatTestingData.^2, 2));
sqrt_flatTrainingData = sqrt(sum(flatTrainingData.^2, 2));
dist_cos = 1 - (flatTestingData * flatTrainingData') ./ (sqrt_flatTestingData * sqrt_flatTrainingData');
[~, sorted_cos] = sort(dist_cos, 2);
nearest_cos = trainingLabel(sorted_cos(:, 1:k));
predict_cos = mode(nearest_cos, 2);
knnCOS_timetaken = toc;
fprintf("COS done\n");

%Ensemble Model implementing
tic;
ensembleModel= fitcensemble(flatTrainingData,trainingLabel,'Method','Bag');
predict_ensemble= predict(ensembleModel,flatTestingData);
ensemble_timetaken=toc;

%Tree Model implementing
tic;
treeModel= fitctree(flatTrainingData,trainingLabel,"AlgorithmForCategorical","OVAbyClass");
predict_tree= predict(treeModel,flatTestingData);
decisiontree_timetaken=toc;

% Calculate accuracy
knnL2_accuracy = sum(predict_euc == testingLabel) / numel(testingLabel);
knnCOS_accuracy = sum(predict_cos == testingLabel) / numel(testingLabel);
ensemble_accuracy = sum(predict_ensemble==testingLabel)/numel(testingLabel);
decisiontree_accuracy = sum(predict_tree==testingLabel)/numel(testingLabel);

fprintf("Euclidean formula time: %.4f s, Accuracy: %.2f%%\n", knnL2_timetaken, knnL2_accuracy * 100);
fprintf("Cosine formula time: %.4f s, Accuracy: %.2f%%\n", knnCOS_timetaken, knnCOS_accuracy * 100);
fprintf("Ensemble time: %.4f s, Accuracy: %.2f%%\n", ensemble_timetaken, ensemble_accuracy * 100);
fprintf("Tree time: %.4f s, Accuracy: %.2f%%\n", decisiontree_timetaken, decisiontree_accuracy * 100);

%save up the confusion matrix
knnL2_confusionmatrix= confusionmat(testingLabel, predict_euc);
knnCOS_confusionmatrix=confusionmat(testingLabel, predict_cos);
ensemble_confusionmatrix=confusionmat(testingLabel, predict_ensemble);
decisiontree_confusionmatrix=confusionmat(testingLabel, predict_tree);

% Plotting the confusion matrix chart
figure;
confusionchart(knnL2_confusionmatrix);
title(sprintf('Euclidean Confusion Matrix.'));

figure;
confusionchart(knnCOS_confusionmatrix);
title(sprintf('Cosine Confusion Matrix.'));

figure;
confusionchart(ensemble_confusionmatrix);
title(sprintf('Ensemble Confusion Matrix.'));

figure;
confusionchart(decisiontree_confusionmatrix);
title(sprintf('Tree Confusion Matrix.'));

%outputting
save("cw1.mat","classes","training_index","decisiontree_timetaken","decisiontree_accuracy","decisiontree_confusionmatrix","ensemble_timetaken","ensemble_accuracy" ...
    ,"ensemble_confusionmatrix","knnCOS_timetaken","knnCOS_accuracy","knnCOS_confusionmatrix","knnL2_timetaken","knnL2_accuracy","knnL2_confusionmatrix")