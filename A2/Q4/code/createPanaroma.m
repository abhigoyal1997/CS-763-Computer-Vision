function p_im = createPanaroma(im, base, thresh)
    % Adding 1st image to panaroma
    p_im = im{base};

    % Adding rest images one by one
    n = length(im);
    for i=1:n
        if i==base
            continue
        end
        [p_pts, i_pts] = featureMatch(p_im, im{i});
        H = ransacHomography(i_pts, p_pts, thresh);
        p_im = stitchImage(im{i},p_im,H);
    end
end