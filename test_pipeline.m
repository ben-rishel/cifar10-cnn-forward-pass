% script for testing CNN pipeline

% load parameters and expected results
load('Project1DataFiles/debuggingTest.mat');
load('Project1DataFiles/CNNparameters.mat');

% compute the CNN output and store it in result
result = cnn_pipeline(imrgb, filterbanks, biasvectors);

% calculate the maximum discrepancy between result and expected values
max(abs(result(:) - layerResults{18}(:)))