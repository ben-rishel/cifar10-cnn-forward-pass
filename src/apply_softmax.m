function outarray = apply_softmax(inarray)
    %inarray is 1x1xD and outarray is the same size
    D = size(inarray, 3); % define dimension size d
    a = max(inarray(1, 1, 1:D)); % compute alpha using max value over all ks
    outarray = zeros(1, 1, D);% initialize outarray
    ksum = 0; % initialize sum of all k values

    % compute sum of shifted exponentials over all k 
    for k = 1:D
        ksum = ksum + exp((inarray(1, 1, k) - a));
    end

    for k = 1:D
        outarray(1,1,k) = (exp(inarray(1, 1, k) - a))/ksum; 
    end
end