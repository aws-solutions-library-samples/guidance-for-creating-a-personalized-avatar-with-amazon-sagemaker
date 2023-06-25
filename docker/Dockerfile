ARG BASE_IMAGE

FROM $BASE_IMAGE

#Install any additional libraries
RUN echo "Adding conda package to Docker image"
RUN mkdir -p /home/condpackenv/

# Copy conda env
COPY stablediff_env.tar.gz /home/condpackenv/stablediff_env.tar.gz

COPY stable_diff.tar.gz /temp/

# Install tar
RUN apt-get update && apt-get install -y tar

# Untar the file
RUN tar -xzf /temp/stable_diff.tar.gz -C /home/

RUN rm /temp/stable_diff.tar.gz