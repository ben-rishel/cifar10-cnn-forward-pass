function outarray = cnn_pipeline(inarray, filterbanks, biasvectors)
% cnn_pipeline applies the 18-layer neural net to a single image
% inarray is a raw image

    % execute CNN pipeline
    imrgb1 = apply_imnormalize(inarray); % 1
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
    outarray = apply_softmax(imrgb17); % 18
end