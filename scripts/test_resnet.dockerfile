FROM pix_base

WORKDIR Pix2PixNIfTI
ADD . .

ENTRYPOINT ["bash", "./scripts/test_resnet.sh"]