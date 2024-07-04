## base container
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

# adding opencontainer labels to link registry to github repository
LABEL org.opencontainers.image.title="prevent-nf"
LABEL org.opencontainers.image.description="PREVENT: PRotein Engineering by Variational frEe eNergy approximaTion"
LABEL org.opencontainers.image.url="https://github.com/stracquadaniolab/prevent-nf"
LABEL org.opencontainers.image.documentation="https://github.com/stracquadaniolab/prevent-nf"
LABEL org.opencontainers.image.source="https://github.com/stracquadaniolab/prevent-nf"
LABEL org.opencontainers.image.vendor="stracquadaniolab"
LABEL org.opencontainers.image.authors="Evgenii Lobzaev"
LABEL org.opencontainers.image.revision="v1.0.0"

# new line (from https://forums.developer.nvidia.com/t/invalid-public-key-for-cuda-apt-repository/212901/13)
# new pubkey
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# added wget, git, gcc, cmake
RUN apt-get update \
    && apt-get install --yes rename procps curl wget git \
    && apt-get install gcc \
    && apt-get -y install cmake protobuf-compiler \
    && apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# copy environment.yml file
WORKDIR /
COPY containers/environment.yml /
RUN conda env update -n base --file environment.yml && conda clean --all --yes

# install tini (and add to the PATH)
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENV PATH="/tini:${PATH}"

ENTRYPOINT ["/tini","-s","--"] 
CMD ["/bin/bash"]
