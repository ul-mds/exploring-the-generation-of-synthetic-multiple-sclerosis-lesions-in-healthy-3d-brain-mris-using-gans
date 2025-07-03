FROM pix_base

WORKDIR Pix2PixNIfTI
ADD . .

ENTRYPOINT ["bash", "./scripts/grid_search_unet.sh"]