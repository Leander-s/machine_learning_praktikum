# Dockerfile
# official miniconda image
FROM continuumio/miniconda3

RUN apt-get update 
    && apt-get install -y git 
    && apt-get install -y vim
    && apt-get clean

RUN git clone https://github.com/Leander-s/machine_learning_praktikum.git

WORKDIR ./machine_learning_praktikum

RUN chmod +x ./init.sh && ./init.sh

CMD ["bash"]
