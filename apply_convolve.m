function outarray = apply_convolve(inarray, filterbank, biasvals)
    %inarray is NxMxD1, filterbank is RxCxD1xD2,
    %biasvals is a length D2 vector, and outarray is NxMxD2
    N = size(inarray, 1);
    M = size(inarray, 2);
    D1 = size(inarray, 3);
    D2 = size(biasvals, 1);

    outarray = zeros(N, M, D2);

    for l = 1:D2
        conv_sum = zeros(N, M);
        for k = 1:D1
            % convolve kth channel of inarray with kth channel of lth filter and add to accumulator
            conv_sum = conv_sum + imfilter(inarray(:,:,k), filterbank(:,:,k,l), 'conv', 'same');
        end
        % add lth bias
        outarray(:,:,l) = conv_sum + biasvals(l);
    end
end