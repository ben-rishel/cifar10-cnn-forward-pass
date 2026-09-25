function outarray = apply_fullconnect(inarray, filterbank, biasvals)
    %inarray is NxMxD1, filterbank is NxMxD1xD2,
    %biasvals is a length D2 vector, and outarray is 1x1xD2
    D2 = size(biasvals, 1);
    outarray = zeros(1, 1, D2);

    for l = 1:D2
        % extract lth filter
        lth_filter = filterbank(:, :, :, l);
        % flatten NxMxD1 arrays to vectors, take dot product, and add bias
        outarray(:,:,l) = dot(inarray(:), lth_filter(:)) + biasvals(l);
    end
end