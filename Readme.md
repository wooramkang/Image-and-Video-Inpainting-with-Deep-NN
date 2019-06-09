
# Video/Image Inpainting NN model
# INIT 2019.05.13.
# deep learning model for Video Inpainting / Image Inpainting

## Dataset 

    * UCF-101
    * https://www.crcv.ucf.edu/data/UCF101.php

## to train image inpainting (frame by frame)

    * run train_image_pdCN.py

    python3 train_image_pdCN.py

## to test image inpainting (frame by frame)

    * run test_pdCN.py

    python3 test_pdCN.py

## to train video inpainting (frames of video)

## to test video inpainting (frames of video)

### Video model is updating


## Model Architecture


### Video Inpainting

    * 3D UNet
    * 2D Unet
    * Combination NN with Temporal 3D NN & Spatial 2D NN
    * etc...

### Image Inpainting

    * PConvolution
    * DliatedConvolution
    * Unet-like 
    * etc...

## reference

1. https://arxiv.org/pdf/1806.08482.pdf

2. https://arxiv.org/pdf/1905.02884.pdf

3. https://arxiv.org/pdf/1711.10925.pdf

4. https://arxiv.org/pdf/1804.07723.pdf

5. https://arxiv.org/pdf/1811.11718.pdf

6. https://arxiv.org/pdf/1810.08771.pdf

7. https://arxiv.org/pdf/1803.02077.pdf

8. https://arxiv.org/pdf/1803.04626.pdf

9. https://arxiv.org/pdf/1903.04227.pdf

10. https://arxiv.org/pdf/1611.09969.pdf

11. https://arxiv.org/pdf/1611.07004.pdf

12. https://arxiv.org/pdf/1808.03344.pdf

13. https://arxiv.org/pdf/1808.06601.pdf

14. https://medium.com/@jonathan_hui/gan-self-attention-generative-adversarial-networks-sagan-923fccde790c

15. https://bluediary8.tistory.com/38

16. https://arxiv.org/abs/1809.11096

17. https://github.com/adamstseng/general-deep-image-completion

18. http://sanghyukchun.github.io/93/

19. https://adjidieng.github.io/Papers/skipvae.pdf

20. http://iizuka.cs.tsukuba.ac.jp/projects/completion/en/

21. https://github.com/lyndonzheng/Pluralistic-Inpainting

22. https://github.com/MingtaoGuo/DCGAN_WGAN_WGAN-GP_LSGAN_SNGAN_RSGAN_BEGAN_ACGAN_PGGAN_TensorFlow

23. https://github.com/MathiasGruber/PConv-Keras

24. https://taeoh-kim.github.io/blog/gan%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-image-to-image-translation-pix2pix-cyclegan-discogan/

25. https://medium.com/@pix2pix.datascience/investigating-the-impact-of-preprocessing-on-image-to-image-translation-a5273d49511e

26. https://medium.com/@jctestud/video-generation-with-pix2pix-aed5b1b69f57

27. https://medium.com/@Synced/nvidia-mit-csail-open-source-video-to-video-synthesis-method-6c876ced2957

28. https://arxiv.org/pdf/1812.09079.pdf

29. https://github.com/flyywh/Video-Super-Resolution

30. https://arxiv.org/pdf/1711.06106.pdf

31. https://arxiv.org/pdf/1806.08482.pdf

32. https://www.profillic.com/paper/arxiv:1901.03419

33. https://www.semanticscholar.org/paper/H-DenseUNet%3A-Hybrid-Densely-Connected-UNet-for-and-Li-Chen/a86d7289c76d832e83c99539859b7b186e4ea6c8

34. https://arxiv.org/abs/1705.00053

35. https://www.offconvex.org/2018/03/12/bigan/

36. https://github.com/SKTBrain/DiscoGAN/blob/master/discogan/model.py

37. https://arxiv.org/pdf/1609.04802.pdf

38. https://wwwpub.zih.tu-dresden.de/~ds24/lehre/ml_ws_2013/ml_11_hinge.pdf
