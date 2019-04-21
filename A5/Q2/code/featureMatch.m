function [p1,p2] = featureMatch(s, t)
    im1 = s(t).cdata;
    im2 = s(t+1).cdata;
    g1 = rgb2gray(im1);
    g2 = rgb2gray(im2);
    
    pts1 = detectSURFFeatures(g1);
    pts2 = detectSURFFeatures(g2);
    [f1, vpts1] = extractFeatures(g1, pts1);
    [f2, vpts2] = extractFeatures(g2, pts2);
    
    idxPairs = matchFeatures(f1, f2);
    p1 = vpts1(idxPairs(:,1));
    p2 = vpts2(idxPairs(:,2));
    
    p1 = p1.Location;
    p2 = p2.Location;
end

