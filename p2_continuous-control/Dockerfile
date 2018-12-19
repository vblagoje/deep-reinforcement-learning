FROM jupyter/tensorflow-notebook:latest
RUN conda install --quiet --yes pytorch torchvision -c soumith
RUN curl https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip -o /home/$NB_USER/Reacher_Linux.zip
RUN unzip /home/$NB_USER/Reacher_Linux.zip
COPY . /home/$NB_USER/
