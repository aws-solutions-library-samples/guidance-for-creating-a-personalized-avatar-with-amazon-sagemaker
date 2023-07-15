ARG BASE_IMAGE

FROM $BASE_IMAGE

#Install any additional libraries
RUN echo "Adding stable diffusion base model to Docker image"
RUN mkdir -p /home/
COPY stable_diff.tar.gz /tmp/
# Install tar
RUN apt-get update && apt-get install -y tar

# Untar the file
RUN tar -xzf /tmp/stable_diff.tar.gz -C /home/

RUN rm /tmp/stable_diff.tar.gz

RUN echo "Install required packages"
COPY requirements.txt /tmp/ 

RUN pip install -r /tmp/requirements.txt
