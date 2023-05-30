from . import audio_util

from torch.utils.data import DataLoader, Dataset, random_split

from .data_loader import df

# ----------------------------
# Sound Dataset
# ----------------------------
class LungSoundDataset(Dataset):
    def __init__(self, df):
        self.df = df

    # ----------------------------
    # Number of items in dataset
    # ----------------------------

    def __len__(self):
        return len(self.df)    
    
    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Get the class id (diagnosis)
        # print(self.df)
        diagnosis = self.df.loc[idx, 'diagnosis']

        sr, raw_sig = audio_util.lungSoundUtil.wavToArray(self.df.loc[idx, 'file'])
        filt_sig = audio_util.lungSoundUtil.filterNoise(sr, raw_sig)
        duration_sig = audio_util.lungSoundUtil.pad_trunc(filt_sig)
        spectogram = audio_util.lungSoundUtil.createMelSpectogram(duration_sig)
        aug_spec = audio_util.lungSoundUtil.augmentSpectogram(spectogram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_spec, diagnosis


lung_sound_ds = LungSoundDataset(df)

# Random split of 80:20 between training and validation
num_items = len(lung_sound_ds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(lung_sound_ds, [num_train, num_val])

# Create training and validation data loaders
train_dl = DataLoader(train_ds, batch_size=num_train, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=num_val, shuffle=False)
