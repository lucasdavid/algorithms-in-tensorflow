"""Download GTA-5 Dataset.

References:
  
  - https://github.com/oscarmcnulty/gta-3d-dataset


"""

import os
import requests
import argparse

SOURCES = [
    'https://s3-us-west-2.amazonaws.com/gtav-captures/67b90283-627b-45cf-9ff2-63dcb95bfc67.zip',
    'https://s3-us-west-2.amazonaws.com/gtav-captures/7007b0bf-503c-4eb7-9b58-19e123ef40e0.zip',
    'https://s3-us-west-2.amazonaws.com/gtav-captures/782579db-da70-492e-a119-4e5bf1241698.zip',
    'https://s3-us-west-2.amazonaws.com/gtav-captures/9bac3205-32d1-4e24-8bc3-7591dbbfac34.zip',
    'https://s3-us-west-2.amazonaws.com/gtav-captures/bcac5255-a6aa-402b-9b75-4d9c422b8ae8.zip',
    'https://s3-us-west-2.amazonaws.com/gtav-captures/e121fb4d-2b4f-40e5-9a34-2658e7647afd.zip',
    'https://s3-us-west-2.amazonaws.com/gtav-captures/e14e4ede-d064-46ae-b513-bab61ca3259f.zip',
    'https://s3-us-west-2.amazonaws.com/gtav-captures/ebecc37a-77ea-46a2-bd54-f67740a411a9.zip',
    'https://s3-us-west-2.amazonaws.com/gtav-captures/ee16f4b5-07f1-4d96-a5b0-92b7de2eee17.zip',
    'https://s3-us-west-2.amazonaws.com/gtav-captures/fd10222e-d26b-4c47-8118-98c8ea545bb4.zip',
    'https://s3-us-west-2.amazonaws.com/gtav-captures/fdf4ad8d-d9b8-49a7-b9a6-c597b8876e0f.zip']


def run(destination):
    for s in SOURCES:
        print('Downloading', s)
        download_file(s, destination)


def download_file(url, destination):
    local_filename = url.split('/')[-1]
    local_filename = os.path.join(destination, local_filename)

    r = requests.get(url, stream=True)

    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def args():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--destination', type=str, default='./data/')

    return p.parse_args()

if __name__ == '__main__':
    namespace = args()
    run(namespace.destination)
