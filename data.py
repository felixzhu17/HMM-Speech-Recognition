fpaths = []
labels = []
spoken = []
for f in os.listdir('audio'):
    for w in os.listdir('audio/' + f):
        fpaths.append('audio/' + f + '/' + w)
        labels.append(f)
        if f not in spoken:
            spoken.append(f)
print('Words spoken:', spoken)

def get_path_mfcc(path):
    y, sr = librosa.load(path, sr = SR)
    return get_mfcc(y, sr, n_mfcc = N_DIMENSIONS)

raw_data = [{"label": label, "mfcc": get_path_mfcc(path)} for path, label in zip(fpaths, labels)]
data = {}
for word in spoken:
    mfcc_samples = [d["mfcc"] for d in raw_data if d["label"] == word]
    data[word] = pad_and_stack(mfcc_samples)

label_map = {label: i for i, label in enumerate(spoken)}
reverse_label_map = {i: label for i, label in enumerate(spoken)}
mfcc_samples, y = [d["mfcc"] for d in raw_data], [label_map[d["label"]] for d in raw_data]
x, mask = pad_and_stack(mfcc_samples)

from sklearn.model_selection import train_test_split

# split data into train and test set
x_train, x_test, y_train, y_test, mask_train, mask_test = train_test_split(x, y, mask, test_size=0.2)

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
class WordDataset(Dataset):
    def __init__(self, x, y, mask):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).long()
        self.mask = torch.tensor(mask)[:, :, 0]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx]

# create dataloaders for train and test set
train_data = DataLoader(WordDataset(x_train, y_train, mask_train), batch_size=32, shuffle=True)
test_data = DataLoader(WordDataset(x_test, y_test, mask_test), batch_size=32, shuffle=True)