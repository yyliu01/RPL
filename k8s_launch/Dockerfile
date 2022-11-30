FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libglib2.0-0 git  && apt-get clean
COPY ../docs/requirements.txt requirements.txt
COPY ../ckpts ckpts
RUN pip install -r requirements.txt
RUN pip install --upgrade google-cloud-storage

# don't change above staffs.
COPY ../rpl_corocl.code rpl_corocl.code
COPY ../rpl.code rpl.code


