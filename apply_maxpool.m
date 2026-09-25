function outarray = apply_maxpool(inarray)
    %inarray is 2Nx2MxD and outarray is size NxMxD
    % extract dimension size
    twoN = size(inarray, 1);
    twoM = size(inarray, 2);
    D = size(inarray, 3);

    % initialize outarray
    outarray = zeros(twoN/2, twoM/2, D);
    
    % iterate over D indices
    for d = 1:D
        % iterate over N indices
        for i = 1:twoN/2
            % iterate over M indices
            for j = 1:twoM/2

                % compute subarray block (using Figure 1A as reference)
                subarray = inarray(2*i-1:2*i, 2*j-1:2*j, d);
                
                % populate outarray with maximum value from vectorized form of subarray 
                outarray(i, j, d) = max(subarray(:));
            end
        end
    end
end