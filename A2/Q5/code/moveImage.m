function retIm = moveImage(im, angle, tx)

retIm = imrotate(im, angle, 'crop');
retIm = imtranslate(retIm, [tx, 0],'FillValues', 0);

retIm(retIm<0) = 0;
retIm(retIm>255) = 255;

