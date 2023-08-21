# Dockerfile to build a minimal 
# ubuntu VM from which to install
# Anaconda Python.

FROM ubuntu:22.04
LABEL org.opencontainers.image.authors="spq@uga.edu"

RUN apt-get update && apt-get -y upgrade
RUN apt-get -y install build-essential
RUN apt-get -y install wget curl vim git

ENTRYPOINT ["/bin/bash"]
