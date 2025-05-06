import numpy as np

def inspect(path):
    X = np.load(f"{path}/X.npy")
    y = np.load(f"{path}/labels.npy")
    print(f"{path} -> X.shape={X.shape}, y.shape={y.shape}, "
          f"classes={np.unique(y)}")
    return X, y

if __name__ == "__main__":
    for ds in [
        r"e:\Exoskeleton_DL\DeepTransferEEG\data\CustomEpoch",
        r"e:\Exoskeleton_DL\DeepTransferEEG\data\BNCI2015001"
    ]:
        inspect(ds)
