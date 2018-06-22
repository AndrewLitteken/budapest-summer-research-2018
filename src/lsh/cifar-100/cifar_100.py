import requests
import shutil
import tarfile
import os

def get_data():
  download_dir = "../../../testing-data/"
  final_path = download_dir + "cifar-100"
  if os.path.exists(final_path):
    return

  download_path = download_dir + "cifar-100.tar.gz"
  download_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
  print("Downloading cifar-100 dataset...")
  r = requests.get(download_url, stream=True)
  with open(download_path, 'wb') as out_file:
    shutil.copyfileobj(r.raw, out_file)
  print("Download Complete")
  
  print("Extracting Cifar-100 dataset")
  tar = tarfile.open(download_path, mode = 'r:gz')

  tar.extractall(path = download_dir)
  print("Extracting Complete")

  os.rename(download_dir +"cifar-100-python", final_path)
  os.remove(download_path)
