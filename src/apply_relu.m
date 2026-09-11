function outarray = apply_relu(inarray)
    %inarray is NxMxD and outarray is the same size
    outarray = inarray;
    outarray(outarray < 0) = 0;
end