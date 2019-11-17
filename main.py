# %% data import
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from pathlib import Path


# %% import data
def load_data(enable_labels):
    train_data = pd.read_csv('titanic/train.csv')
    # test_data = pd.read_csv('titanic/test.csv')

    train_x = train_data[enable_labels]
    # test_x = test_data[enable_labels]

    train_y = train_data['Survived']
    # test_y = test_data['Survived']

    return train_x, train_y


# %%
def null_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum() / len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
        columns={0: '欠損数', 1: '%'})
    return kesson_table_ren_columns


# %%
class LNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LNN, self).__init__()
        DEPTH = 16
        self.fc_input = nn.Linear(input_dim, 256)
        self.fc_array = nn.ModuleList([nn.Linear(256, 256) for _ in range(DEPTH - 2)])
        self.fc_output = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.dropout(x, training=self.training)
        y = F.relu(self.fc_input(x))
        for layer in self.fc_array:
            y = F.dropout(y, training=self.training)
            y = F.relu(layer(y))
        y = F.dropout(y, training=self.training)
        y = self.fc_output(y)
        return y

    def pred(self, x):
        x = torch.from_numpy(x).float()
        return self.forward(x).detach().numpy()

    def save(self, save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), f'{save_path}/lnn.pth')

    def load(self, load_path):
        self.load_state_dict(torch.load(f'{load_path}/lnn.pth'))


class BNLNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BNLNN, self).__init__()
        DEPTH = 16
        self.fc_input = nn.Linear(input_dim, 256)
        self.fc_array = nn.ModuleList([nn.Linear(256, 256) for _ in range(DEPTH - 2)])
        self.bn_array = nn.ModuleList([nn.BatchNorm1d(256) for _ in range(DEPTH - 2)])
        self.fc_output = nn.Linear(256, output_dim)

    def forward(self, x):
        y = F.relu(self.fc_input(x))
        for layer, bn in zip(self.fc_array, self.bn_array):
            y = F.relu(bn(layer(y)))
        y = self.fc_output(y)
        return y

    def pred(self, x):
        x = torch.from_numpy(x).float()
        return self.forward(x).detach().numpy()

    def save(self, save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), f'{save_path}/bnlnn.pth')

    def load(self, load_path):
        self.load_state_dict(torch.load(f'{load_path}/bnlnn.pth'))


def get_null_index(data):
    null_index = data.isnull().any(axis=1)
    return null_index


def preprocess_for_trainingdata(train_x, train_y):
    # とりあえず一つでも欠損していればそのデータは有効にしないようにする
    # train_x = train_x.values
    null_index = get_null_index(train_x)
    train_x = train_x[~null_index]
    train_y = train_y[~null_index]

    MALE = .0
    FEMALE = 1.0
    Q = .0
    S = 1.0
    C = 2.0

    train_x = train_x.replace('male', MALE)
    train_x = train_x.replace('female', FEMALE)
    train_x = train_x.replace('Q', Q)
    train_x = train_x.replace('S', S)
    train_x = train_x.replace('C', C)

    return train_x, train_y


def train(model, loss_func, optimizer, trX, trY):
    x = Variable(trX, requires_grad=False)
    y = Variable(trY, requires_grad=False)
    optimizer.zero_grad()
    y_pred = model.forward(x)
    loss = loss_func(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.data


def training(model, train_x, train_y, epochs, batch_size, save_path, eval_num=20):

    train_ = data.TensorDataset(torch.from_numpy(train_x).float(),
                                torch.from_numpy(train_y).float())
    train_iter = torch.utils.data.DataLoader(train_, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    torch.manual_seed(1)
    for epoch in range(1, epochs + 1):
        model.train()
        loss = 0
        for i, train_data in enumerate(train_iter):
            inputs, labels = train_data
            loss += train(model, criterion, optimizer, inputs, labels)
        print(f'epoch {epoch}: loss {loss / batch_size}')

        if epoch % eval_num == 0:
            model.eval()
            eval = model.pred(train_x)
            eval = np.round(eval)
            y = np.reshape(train_y, eval.shape)
            res = np.abs(eval - y)
            ratio = 1.0 - np.sum(res, dtype=np.float32) / len(train_y)
            print(f'{epoch} test percentage {ratio}')

            model.save(save_path)
            print(f'save model at {save_path}')


def training_LNN():
    train_x, train_y = load_data(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
    train_x, train_y = preprocess_for_trainingdata(train_x, train_y)
    model = LNN(7, 1)
    training(model, train_x.values, train_y.values, 500, 32, 'models/lnn/', eval_num=2)


def training_BNLNN():
    train_x, train_y = load_data(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
    train_x, train_y = preprocess_for_trainingdata(train_x, train_y)
    model = BNLNN(7, 1)
    training(model, train_x.values, train_y.values, 500, 32, 'models/bnlnn/', eval_num=2)


if __name__ == '__main__':
    training_BNLNN()
