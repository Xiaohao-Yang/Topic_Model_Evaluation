from torch.utils.data import Dataset


class FeatDataset(Dataset):
    def __init__(self, data_np, ):
        self.data = data_np
        self.word_count = data_np.sum(1)

    def __getitem__(self, item):
        return self.data[item], self.word_count[item]

    def __len__(self):
        return len(self.data)