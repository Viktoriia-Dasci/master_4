import zipfile
with zipfile.ZipFile("/home/viktoriia.trokhova/Slices.zip","r") as zip_ref:
    zip_ref.extractall("/home/viktoriia.trokhova/Slices")
