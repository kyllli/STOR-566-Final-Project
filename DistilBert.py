# Reference: https://www.kaggle.com/code/samson22/distilbert-in-pytorch
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertModel

MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-05
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

train_data = pd.read_csv('df.csv')

label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_data['labels'] = train_data[label_columns].apply(lambda x: list(x), axis=1)

train_data.drop(['id'], inplace=True, axis=1)
train_data.drop(label_columns, inplace=True, axis=1)

class MultiLabelDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, new_data=False):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.comment_text
        self.new_data = new_data
        
        if not new_data:
            self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        out = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }
        
        if not self.new_data:
            out['targets'] = torch.tensor(self.targets[index], dtype=torch.float)

        return out

train_size = 0.9

train_df = train_data.sample(frac=train_size, random_state=22)
val_df = train_data.drop(train_df.index).reset_index(drop=True)
train_df = train_df.reset_index(drop=True)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
training_set = MultiLabelDataset(train_df, tokenizer, MAX_LEN)
val_set = MultiLabelDataset(val_df, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 8
                }

val_params = {'batch_size': VALID_BATCH_SIZE,
               'shuffle': False,
               'num_workers': 8
                }

training_loader = DataLoader(training_set, **train_params)

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, 6)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        out = hidden_state[:, 0]
        out = self.classifier(out)
        return out

model = DistilBERTClass()
model.to(DEVICE);
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

def train(epoch=1):
    model.train()
    
    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(DEVICE, dtype=torch.long)
        mask = data['mask'].to(DEVICE, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(DEVICE, dtype=torch.long)
        targets = data['targets'].to(DEVICE, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
        
        if _ % 5000 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        loss.backward()
        optimizer.step()

for epoch in range(EPOCHS):
    train(epoch)

test_set = MultiLabelDataset(val_set, tokenizer, MAX_LEN, new_data=True)
test_loader = DataLoader(val_set, **val_params)

all_test_pred = []
loss_val=0

def test(epoch):
    model.eval()
    
    with torch.inference_mode():
    
        for _, data in tqdm(enumerate(test_loader, 0)):
            ids = data['ids'].to(DEVICE, dtype=torch.long)
            mask = data['mask'].to(DEVICE, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(DEVICE, dtype=torch.long)
            targets = data['targets'].to(DEVICE, dtype=torch.float)
            
            outputs = model(ids, mask, token_type_ids)
            probas = torch.sigmoid(outputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
            accuracy+=(outputs==targets)
            loss_val+=loss
            all_test_pred.append(probas)
    return probas, accuracy,loss_val
probas, accuracy, loss_val = test(model)
print("accuracy is equal to", accuracy/15957,"the loss is equal to",loss_val/15957)

