% script for running the CNN on all 10,000 images

% make functions in the 'src' directory callable
addpath('src');

% load test data and parameters
load('Project1DataFiles/cifar10testdata.mat');
load('Project1DataFiles/CNNparameters.mat');

% initialize 10x10 confusion matrix
confusion_matrix = zeros(10, 10);

for i = 1:10000
    % get the i-th image
    img = imageset(:,:,:,i);

    % run it through the pipeline
    probs = cnn_pipeline(img, filterbanks, biasvectors);
    probs = squeeze(probs); % converting to plain array

    % find the predicted class
    [maxprob, predicted] = max(probs);

    % update the confusion matrix
    confusion_matrix(trueclass(i), predicted) = confusion_matrix(trueclass(i), predicted) + 1;
end

% calculate accuracy
accuracy = sum(diag(confusion_matrix)) / sum(sum(confusion_matrix));

% display confusion matrix
fprintf('Confusion Matrix:\n');
disp(confusion_matrix);

% print accuracy
fprintf('Classification Accuracy: %.4f\n', accuracy);