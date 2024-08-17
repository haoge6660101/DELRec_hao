import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from SR_models import GRU


class UserInteractionDataset(Dataset):
    def __init__(self, interactions, item_num):
        self.interactions = interactions
        self.item_num = item_num

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user_interactions = self.interactions[idx]
        return torch.tensor(user_interactions[:-1], dtype=torch.long), torch.tensor(user_interactions[-1],
                                                                                    dtype=torch.long), len(
            user_interactions) - 1


user_interactions = pd.read_csv('../data_user_interactions.csv')
titles = pd.read_csv('../title_set.csv')

item_to_title = dict(zip(titles['item_id'], titles['title']))
user_interactions['interactions'] = user_interactions['interactions'].apply(
    lambda x: [int(i.split('_')[1]) for i in x.split(',')])

hidden_size = 64
item_num = len(item_to_title)
state_size = 5
gru_layers = 3
dropout_rate = 0.3
k = 10  # or 15

dataset = UserInteractionDataset(user_interactions['interactions'].tolist(), item_num)
dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

model = GRU(hidden_size, item_num, state_size, gru_layers)
optimizer = optim.Adagrad(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(600):
    for states, labels, len_states in dataloader:
        optimizer.zero_grad()
        outputs = model(states, len_states)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

model.eval()
recommendations = []
with torch.no_grad():
    for states, _, len_states in dataloader:
        outputs = model(states, len_states)
        _, top_k_items = torch.topk(outputs, k, dim=1)
        recommendations.extend(top_k_items.tolist())


def insert_recommendations(interactions, recommendations, k):
    return interactions[:-1] + recommendations[:k] + [interactions[-1]]


user_interactions['recommendations'] = recommendations
user_interactions['recommendations'] = user_interactions['recommendations'].apply(lambda x: [f'item_{i}' for i in x])
user_interactions['all_interactions'] = user_interactions.apply(
    lambda row: insert_recommendations(row['interactions'], row['recommendations'], k), axis=1)
user_interactions['all_interactions'] = user_interactions['all_interactions'].apply(
    lambda x: [item_to_title[f'item_{i}'] for i in x])

user_interactions.to_csv('../user_interactions_with_text_title_and_predicted_items_by_GRU.csv', index=False)
with open('../user_interactions_with_text_title_and_predicted_items_by_GRU.txt', 'w') as f:
    for index, row in user_interactions.iterrows():
        f.write(','.join(row['all_interactions']) + '\n')
