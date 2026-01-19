FROM pix_base

WORKDIR Pix2PixNIfTI
ADD . .

ENTRYPOINT ["bash", "./scripts/val_unet.sh"]