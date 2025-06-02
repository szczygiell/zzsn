import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import MBart50TokenizerFast


class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.embedding = nn.Embedding(tokenizer.vocab_size, 256)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.data.iloc[idx]['english']
        tgt_text = self.data.iloc[idx]['polish']

        src = self.tokenizer(src_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        tgt = self.tokenizer(tgt_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)

        src_ids = src['input_ids'].squeeze(0)#.float()  # Placeholder; should embed properly
        tgt_ids = tgt['input_ids'].squeeze(0)#.float()

        src_emb = self.embedding(src_ids)
        tgt_emb = self.embedding(tgt_ids)

        return src_emb, tgt_emb


class RLST(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.embedding_in = nn.Linear(input_dim, hidden_dim)
        self.embedding_out = nn.Linear(output_dim, hidden_dim)
        self.rnn = nn.GRU(2*hidden_dim, hidden_dim, num_layers=4, dropout=0.2, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5)
        )
        self.token_output = nn.Linear(hidden_dim, output_dim)
        self.q_write = nn.Linear(hidden_dim, 1)
        self.q_read = nn.Linear(hidden_dim, 1)

    def forward(self, x_token, y_token, hidden):
        x = self.embedding_in(x_token)
        y = self.embedding_out(y_token)
        combined = torch.cat([x, y], dim=-1)  # Shape: [batch, hidden]
        out, hidden = self.rnn(combined.unsqueeze(0).unsqueeze(0), hidden)
        out = self.dense(out.squeeze(1))
        token_pred = self.token_output(out)
        q_w = self.q_write(out)
        q_r = self.q_read(out)
        return token_pred, q_w, q_r, hidden


def select_action(q_w, q_r, epsilon=0.3):
    if random.random() < epsilon:
        return random.choice(["READ", "WRITE"])
    return "WRITE" if q_w > q_r else "READ"


def compute_target(q_next_w, q_next_r, reward, done, gamma=0.9):
    if done:
        return reward
    return reward + gamma * max(q_next_w, q_next_r)


def train_step(model, optimizer, x_seq, y_seq, loss_fn_ce, loss_fn_mse, device):
    model.train()
    hidden = None
    loss_m, loss_e = 0.0, 0.0

    i, j = 0, 0
    x_item = x_seq[i]
    if isinstance(x_item, torch.Tensor):
        x_token = x_item.detach().clone().to(torch.float32).to(device)
    else:
        x_token = torch.tensor(x_item, dtype=torch.float32).to(device)
        print('x_token nie tensor')

    y_item = y_seq[0]
    if isinstance(y_item, torch.Tensor):
        y_token = torch.zeros_like(y_item.detach().clone()).to(torch.float32).to(device)
    else:
        y_token = torch.zeros_like(torch.tensor(y_item, dtype=torch.float32)).to(device)
        print('y_token nie tensor')

    for step in range(len(x_seq) + len(y_seq)):
        token_pred, q_w, q_r, hidden = model(x_token, y_token, hidden)
        action = select_action(q_w.item(), q_r.item())

        if action == "READ" and i < len(x_seq):
            x_item = x_seq[i]
            if isinstance(x_item, torch.Tensor):
                next_x_token = x_item.detach().clone().to(torch.float32).to(device)
            else:
                next_x_token = torch.tensor(x_item, dtype=torch.float32).to(device)
                print('x_token_next nie tensor')
            next_y_token = y_token
            reward = 0.0
            done = False
            i += 1
        elif action == "WRITE" and j < len(y_seq):
            next_x_token = x_token
            y_item = y_seq[j]
            if isinstance(y_item, torch.Tensor):
                next_y_token = y_item.detach().clone().to(torch.float32).to(device)
            else:
                next_y_token = torch.tensor(y_item, dtype=torch.float32).to(device)
                print('y_token_next nie tensor')
            reward = -loss_fn_ce(token_pred, next_y_token.unsqueeze(0))
            done = (j == len(y_seq) - 1)
            j += 1
        else:
            reward = -3.0  # M penalty
            next_x_token = x_token
            next_y_token = y_token
            done = True

        with torch.no_grad():
            next_pred, next_q_w, next_q_r, _ = model(next_x_token, next_y_token, hidden)
            q_target = compute_target(next_q_w.item(), next_q_r.item(), reward, done)

        if action == "READ":
            q_est = q_r
        else:
            q_est = q_w
            loss_m += reward ** 2  # surrogate for token prediction loss

        loss_e += loss_fn_mse(q_est, torch.tensor([[q_target]], device=device))

        x_token = next_x_token
        y_token = next_y_token

    loss = loss_m + loss_e
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()

    return loss.item()


def translate(model, input_seq, device, max_len=50):
    model.eval()
    hidden = None
    outputs = []

    x_i = 0
    x_token = torch.tensor(input_seq[x_i], dtype=torch.float32).to(device)
    y_token = torch.zeros_like(x_token).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            token_pred, q_w, q_r, hidden = model(x_token, y_token, hidden)
            action = "WRITE" if q_w > q_r else "READ"

            if action == "READ" and x_i < len(input_seq) - 1:
                x_i += 1
                x_token = torch.tensor(input_seq[x_i], dtype=torch.float32).to(device)
            elif action == "WRITE":
                pred_token = token_pred.squeeze().cpu().numpy()
                outputs.append(pred_token)
                y_token = token_pred.squeeze()
            else:
                break
    return outputs