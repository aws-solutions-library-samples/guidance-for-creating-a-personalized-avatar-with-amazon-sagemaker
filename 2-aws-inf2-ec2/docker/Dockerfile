ARG BASE_IMAGE

FROM $BASE_IMAGE

# Copy bash script into image 
RUN echo "Copy bash script into image "

# Install python requirements
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY setup.sh /setup.sh

# Make script executable and run it
RUN chmod +x /setup.sh
RUN source /setup.sh -inf2 -p