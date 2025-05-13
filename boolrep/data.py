import lightning as L
from torch import utils
from transformers import AutoTokenizer
import json
        
class BooleanQueryDataset(utils.data.Dataset):
    def __init__(self, tokenizer, max_length=200):
        super().__init__()
        self.q, self.d = self.fetch_dataset()

        assert tokenizer is not None
        self.tokenizer = tokenizer

        self.tokenized_d = tokenizer(
            list(self.d),
            padding=True, 
            truncation=True, 
            max_length=max_length,
            return_tensors='pt'            
        )
        self.tokenized_q = tokenizer(
            list(self.q), 
            padding=True, 
            truncation=True, 
            max_length=max_length,
            return_tensors='pt'
        )

    def fetch_dataset(self):
        d, q = [], []
        with open("data/training.jsonl", "r") as f:
            for line in f:
                x = json.loads(line)  
                d.append(x["d"])
                q.append(x["q"])
        return d, q

    def __len__(self):
        return len(self.q)

    def __getitem__(self, index):
        d_item = {
            key+"_d": values[index]
            for key, values in self.tokenized_d.items()
        }
        q_item = {
            key+"_q": values[index]
            for key, values in self.tokenized_q.items()
        }
        return d_item | q_item

class BooleanQueryDataModule(L.LightningDataModule):
    def __init__(self, model_name):
        super().__init__()

        self.val_split = 0.1
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        pass

    @staticmethod
    def split_data(dataset: BooleanQueryDataset, val_split: float):
        train_length = int((1 - val_split) * len(dataset))
        val_length = len(dataset) - train_length
        train_dataset, val_dataset = utils.data.random_split(
            dataset, lengths=[train_length, val_length]
        )
        return train_dataset, val_dataset
    
    def setup(self, stage):
        self.train_dataset, self.val_dataset = self.split_data(
            BooleanQueryDataset(tokenizer=self.tokenizer),
            val_split=self.val_split
        )

    def train_dataloader(self):
        return utils.data.DataLoader(
            self.train_dataset,
            batch_size=16,
            num_workers=4,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return utils.data.DataLoader(
            self.val_dataset,
            batch_size=16,
            num_workers=4,
            persistent_workers=True,
        )
