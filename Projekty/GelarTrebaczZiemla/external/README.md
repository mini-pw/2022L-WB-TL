In this folder we store files that we use with other repositories.

## External repo 1:

pytorch-CycleGAN-and-pix2pix

clone command: ```git clone git@github.com:junyanz/pytorch-CycleGAN-and-pix2pix.git```

In order to use those files go to external folder in root directory of the project. Then clone using above command. After that copy files from ``src/external/pytorch-CycleGAN-and-pix2pix`` to ``external/pytorch-CycleGAN-and-pix2pix``
to corresponding folders. 

Example commands to run training:

- just GAN: ```python train.py --dataroot./datasets/heart --name heart_cyclegan --model cycle_gan --dataset_mode=heart_gan --lambda_A=10 --lambda_B=10 --lambda_identity=0 --batch_size=8 --input_nc=1 --output_nc=1 --num_threads=12 --gan_mode=vanilla --load_size=128 --netG=resnet_6blocks```
- whole network: ```python train.py --dataroot ./datasets/heart --name heart_mutual --model mutual --dataset_mode=heart_mutual --lambda_A=10 --lambda_T=10 --batch_size=1 --input_nc=1 --output_nc=1 --num_threads=12 --gan_mode=vanilla --load_size=96 --netG=resnet_6blocks```


