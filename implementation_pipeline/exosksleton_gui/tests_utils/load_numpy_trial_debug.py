import numpy 

#load a numpy trial file for debugging and print its shape

def load_numpy_trial_debug(file_path):
    data = numpy.load(file_path)
    print(f"Loaded data shape: {data.shape}")

    return data

if __name__ == "__main__":
    path = r"E:\Exoskeleton_DL\DeepTransferEEG\testt\uu_1_20250618_161656\trial_3_fixation.npy"
    load_numpy_trial_debug(path)