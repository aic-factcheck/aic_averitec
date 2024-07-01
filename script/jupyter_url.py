import glob
import os
import re

PATH = "logs/"

list_of_files = glob.glob(f"{PATH}*")
latest_file = max(list_of_files, key=os.path.getctime)
file_with_largest_no = max(list_of_files, key=lambda x: int(x.split("/")[-1].split(".")[-2]))


with open(file_with_largest_no) as f:
    for line in f:
        matches = re.findall(r"(https?://\S+)", line)
        if matches:
            print(matches[0])
            break
