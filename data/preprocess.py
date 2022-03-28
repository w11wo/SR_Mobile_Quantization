import glob
import shutil
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def main():
    jpg_path = "JPG 25%"
    ori_path = "Images Original"

    get_id = lambda path: path.split(".")[0].split("/")[-1]

    jpg_ids = list(map(get_id, glob.glob(f"{jpg_path}/*")))
    ori_ids = list(map(get_id, glob.glob(f"{ori_path}/*.jpg")))
    ids = sorted(list(set(jpg_ids) & set(ori_ids)))

    df = pd.DataFrame({"id": ids})
    train, val = train_test_split(df, test_size=0.1, random_state=42)

    save_path = "SR_Mobile_Quantization/data"

    with open(f"{save_path}/train.txt", "w") as f:
        for item in train["id"].tolist():
            f.write(f"{item}.pt\n")

    with open(f"{save_path}/val.txt", "w") as f:
        for item in val["id"].tolist():
            f.write(f"{item}.pt\n")

    os.makedirs(f"{save_path}/train_LR", exist_ok=True)
    os.makedirs(f"{save_path}/train_HR", exist_ok=True)

    for id in ids:
        shutil.move(f"{jpg_path}/{id}.jpg", f"{save_path}/train_LR/{id}.jpg")
        shutil.move(f"{ori_path}/{id}.jpg", f"{save_path}/train_HR/{id}.jpg")


if __name__ == "__main__":
    main()
