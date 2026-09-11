function outarray = apply_imnormalize(inarray)
    %inarray is an NxMx3 uint8 image and outarray is NxMx3
    N = size(inarray, 1);
    M = size(inarray, 2);

    outarray = zeros(N, M , 3);

    mu_R = 0.4914;
    mu_G = 0.4822;
    mu_B = 0.4465;

    sigma_R = 0.2470;
    sigma_G = 0.2435;
    sigma_B = 0.2616;

    tmp = double(inarray) / 255.0;

    outarray(:, :, 1) = (tmp(:, :, 1) - mu_R) / sigma_R;
    outarray(:, :, 2) = (tmp(:, :, 2) - mu_G) / sigma_G;
    outarray(:, :, 3) = (tmp(:, :, 3) - mu_B) / sigma_B;
end