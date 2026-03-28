#!/usr/bin/env python3
import kagglehub
import zipfile
import os

zip_path = kagglehub.dataset_download("debashishsau/aslamerican-sign-language-aplhabet-dataset", output_dir="./data/", force_download=True)

# print("Unzipping dataset")
# extract_dir = "./data/ASL_Alphabet_Dataset"
# os.makedirs(extract_dir, exist_ok=True)
# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_dir)
# print("Done")
