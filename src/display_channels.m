function display_channels(inarray, figtitle)
    %inarray is NxMxD, each of the D channels is shown as a greyscale image
    %figtitle is a string used as the figure title
    D = size(inarray, 3);
    cols = ceil(sqrt(D));
    rows = ceil(D / cols);

    figure;
    for k = 1:D
        subplot(rows, cols, k);
        imagesc(inarray(:,:,k));
        colormap(gca, 'gray');
        axis image off;
        title(sprintf('ch %d', k));
    end
    sgtitle(figtitle);
end
