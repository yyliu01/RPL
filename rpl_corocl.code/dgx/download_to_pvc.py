import logging
import os
import zipfile
from pathlib import Path
from typing import Union
from google.cloud import storage


# log = logging.getLogger(__file__)
# log.setLevel(logging.DEBUG)

bucket_namespace = ""
bucket_name = ""


def get_bucket(bucket_namespace: str, bucket_name: str):
    client = storage.Client(project=bucket_namespace)
    bucket = client.get_bucket(bucket_name)
    return bucket


def download_ex5_dataset_unzip(data_dir: str, prefix, bucket_prefix, pvc=False):
    print(os.listdir('/'))
    if pvc:
        data_dir = "/pvc/" + data_dir
    dst_folder = Path(data_dir)
    print('Destination Foder list ==> {}'.format(dst_folder))
    if dst_folder.exists() and pvc:
        print('Skipping download as data dir already exists')
        return
    else:
        print('searching blob ...')
        bucket = get_bucket('', '')
        blob = bucket.blob(prefix+bucket_prefix)
        print('downloading ...')
        path = "/pvc" if pvc else "./"
        with open(Path(path)/bucket_prefix, 'wb') as sink:
            blob.download_to_file(sink)
        print('unziping the {}/{} ...'.format(path, bucket_prefix))
        print(Path(path))
        with zipfile.ZipFile('{}/{}'.format(path, bucket_prefix), 'r') as zip_ref:
            zip_ref.extractall('{}/'.format(path))
    print(os.listdir(path))


def upload_checkpoint(local_path: str, prefix: str, checkpoint_filepath: Union[Path, str]):
    """Upload a model checkpoint to the specified bucket in GCS."""
    bucket_prefix = prefix
    src_path = f"{local_path}/{checkpoint_filepath}"
    dst_path = f"{bucket_prefix}/{checkpoint_filepath}"
    print('Uploading {} => {}'.format(src_path, dst_path))
    # print(os.path.exists(src_path))
    bucket = get_bucket(bucket_namespace, bucket_name)
    # print('searching blob ...')
    blob = bucket.blob(dst_path)
    # print('start uploading ...')
    blob.upload_from_filename(src_path)
    print('finish uploading.')


def download_checkpoint(checkpoint_filepath: str, prefix: str, bucket_namespace= '', bucket_name= ''):
    src_path = f"yy/exercise_5/{prefix}"
    dest_path = f"{checkpoint_filepath}/{prefix}"
    print('Downloading {} => {}'.format(src_path, checkpoint_filepath))
    bucket = get_bucket(bucket_namespace, bucket_name)
    print('searching blob ...')
    blob = bucket.blob(src_path)
    print('start downloading ...')
    
    blob.download_to_filename(dest_path)
    print('finish downloading.')



