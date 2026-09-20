% demo script that visualizes intermediate and final results of the CNN

% make functions in the 'src' directory callable
addpath('src');

% load parameters and a labeled test image
load('Project1DataFiles/CNNparameters.mat');
load('Project1DataFiles/cifar10testdata.mat');

% pick one sample image to run through the pipeline
imgindex = 1;
img = imageset(:,:,:,imgindex);

% show the input image
figure;
imagesc(img);
axis image off;
title(sprintf('input image (true class: %s)', classlabels{trueclass(imgindex)}));

% run the CNN one layer at a time, keeping every intermediate result
imrgb1 = apply_imnormalize(img); % 1
imrgb2 = apply_convolve(imrgb1, filterbanks{2}, biasvectors{2}); % 2
imrgb3 = apply_relu(imrgb2); % 3
imrgb4 = apply_convolve(imrgb3, filterbanks{4}, biasvectors{4}); % 4
imrgb5 = apply_relu(imrgb4); % 5
imrgb6 = apply_maxpool(imrgb5); % 6
imrgb7 = apply_convolve(imrgb6, filterbanks{7}, biasvectors{7}); % 7
imrgb8 = apply_relu(imrgb7); % 8
imrgb9 = apply_convolve(imrgb8, filterbanks{9}, biasvectors{9}); % 9
imrgb10 = apply_relu(imrgb9); % 10
imrgb11 = apply_maxpool(imrgb10); % 11
imrgb12 = apply_convolve(imrgb11, filterbanks{12}, biasvectors{12}); % 12
imrgb13 = apply_relu(imrgb12); % 13
imrgb14 = apply_convolve(imrgb13, filterbanks{14}, biasvectors{14}); % 14
imrgb15 = apply_relu(imrgb14); % 15
imrgb16 = apply_maxpool(imrgb15); % 16
imrgb17 = apply_fullconnect(imrgb16, filterbanks{17}, biasvectors{17}); % 17
imrgb18 = apply_softmax(imrgb17); % 18

% display feature maps at a few representative layers
display_channels(imrgb2, 'layer 2 output (conv, 32x32x10)');
display_channels(imrgb4, 'layer 4 output (conv, 32x32x10)');
display_channels(imrgb6, 'layer 6 output (maxpool, 16x16x10)');
display_channels(imrgb9, 'layer 9 output (conv, 16x16x10)');
display_channels(imrgb11, 'layer 11 output (maxpool, 8x8x10)');
display_channels(imrgb16, 'layer 16 output (maxpool, 4x4x10)');

% display final class probabilities as a bar chart
probs = squeeze(imrgb18);
figure;
bar(probs);
set(gca, 'xtick', 1:10, 'xticklabel', classlabels);
xtickangle(45);
ylabel('probability');
title('layer 18 output (softmax class probabilities)');

% report the predicted class
[maxprob, predicted] = max(probs);
fprintf('predicted class: %s (probability %.4f)\n', classlabels{predicted}, maxprob);
fprintf('true class: %s\n', classlabels{trueclass(imgindex)});
