function entro = getJointEntropy(im1, im2)

arr_im1 = ceil((im1(:)+1)/10);
arr_im2 = ceil((im2(:)+1)/10);
histvalues = zeros(ceil(255/10), ceil(255/10));
for i=1:length(arr_im1)
    histvalues(arr_im1(i), arr_im2(i)) = histvalues(arr_im1(i), arr_im2(i)) + 1;
end
histvalues = histvalues/sum(histvalues(:));
entro = -1*sum(histvalues(histvalues>0).*log(histvalues(histvalues>0)));