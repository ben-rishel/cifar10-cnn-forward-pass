% script for computing and plotting top-k classification accuracy
% (extra credit: accuracy vs. k, for k = 1..10)

% make functions in the 'src' directory callable
addpath('src');

% load test data and parameters
load('Project1DataFiles/cifar10testdata.mat');
load('Project1DataFiles/CNNparameters.mat');

numImages = 10000;
numClasses = 10;

% rank_counts(r) = number of images where the true class was the
% r-th highest scoring class (rank 1 = highest probability)
rank_counts = zeros(numClasses, 1);

for i = 1:numImages
    % get the i-th image
    img = imageset(:,:,:,i);

    % run it through the pipeline
    probs = cnn_pipeline(img, filterbanks, biasvectors);
    probs = squeeze(probs); % converting to plain array

    % sort class indices by probability, highest first
    [~, sortedClasses] = sort(probs, 'descend');

    % find where the true class landed in the sorted ranking
    rank = find(sortedClasses == trueclass(i));

    % record it
    rank_counts(rank) = rank_counts(rank) + 1;
end

% top-k accuracy is the cumulative fraction of images whose true class
% appeared at rank <= k
accuracy_by_k = cumsum(rank_counts) / numImages * 100;

% display the numbers
fprintf('Top-k Accuracy:\n');
for k = 1:numClasses
    fprintf('  k=%2d: %.2f%%\n', k, accuracy_by_k(k));
end

% plot the curve
figure;
plot(1:numClasses, accuracy_by_k, '-o', 'LineWidth', 1.5);
xlabel('k');
ylabel('Accuracy (%)');
title('Top-k Classification Accuracy');
xlim([1 numClasses]);
ylim([0 100]);
xticks(1:numClasses);
grid on;
