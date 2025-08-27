FROM nvcr.io/nvidia/pytorch:21.02-py3

# Fix for running cv2: ffmpeg libsm6 libxext6
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN conda install -y matplotlib numpy=>1.19.2 Pillow>=8.0.1 && \
    conda install -y scipy six && \
    conda install visdom dominate nibabel -c conda-forge && \
    pip install --no-cache-dir opencv-python torchsummary torchio==0.17.50 && \
    conda clean --yes --all