import torch
import pandas as pd
from embedding import get_essay_dataset
from config import config
from torch.utils.data import Dataset, DataLoader
import os
import time
import torch.nn as nn
import numpy as np
import gc
from sklearn.metrics import cohen_kappa_score, accuracy_score
from psutil import virtual_memory
import os, numpy as np
import dask.dataframe as dd
import torch.optim as optim

class GRUScoreModule(nn.Module):
    def __init__(self, output_dim, hidden_dim, dropout=0.5):
        super(GRUScoreModule, self).__init__()
        # GRU layer
        self.gru = nn.GRU(768, hidden_dim, num_layers=3, dropout=dropout, 
                          batch_first=True, bidirectional=True)
        
        # Layer Normalization 추가
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Dropout and FC layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # GRU forward pass
        x, _ = self.gru(x)
        
        # Use the output of the last time step
        x = x[:, -1, :]
        
        # Apply layer normalization (개선점)
        x = self.layer_norm(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Final FC layer
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x


class EssayDataset(Dataset):
    def __init__(self, embedded_essays, labels):
        # 원본 데이터를 그대로 저장 (패딩하지 않음)
        self.embedded_essays = embedded_essays
        # labels만 텐서로 변환
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.maxlen = 128

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 필요할 때마다 패딩 및 텐서 변환 수행
        essay = self.embedded_essays[idx]
        
        # numpy array로 변환 (DataFrame인 경우)
        if hasattr(essay, 'values'):
            essay = essay.values
        
        # 패딩 수행
        if len(essay) < self.maxlen:
            # pre-padding (앞쪽에 0 추가)
            padded = np.zeros((self.maxlen, essay.shape[1]), dtype=np.float32)
            padded[-len(essay):] = essay
        else:
            # maxlen까지만 자르기
            padded = essay[:self.maxlen].astype(np.float32)
        
        # 텐서로 변환
        padded_tensor = torch.tensor(padded, dtype=torch.float32)
        
        return padded_tensor, self.labels[idx]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_sent_pred, y_test, args):
    metrics = {}
    all_kappas = []
    for i in range(len(args['rubric'])):
        metrics[args['rubric'][i]] = {}
        y_pred = y_sent_pred[:, i]
        y_true = y_test[:, i]
        accuracy = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        metrics[args['rubric'][i]]['accuracy'] = accuracy
        metrics[args['rubric'][i]]['kappa'] = kappa
        all_kappas.append(kappa)

    metrics['mean'] = {}
    overall_accuracy = accuracy_score(y_test.flatten(), y_sent_pred.flatten())
    overall_kappa = np.mean(all_kappas)
    metrics['mean']['accuracy'] = overall_accuracy
    metrics['mean']['kappa'] = overall_kappa

    metrics['overall'] = {}
    metrics['overall']['kappa'] = cohen_kappa_score(y_test.flatten(), y_sent_pred.flatten(), weights='quadratic')
    return metrics
    
def get_embedded_essay(args, valid_index, essays):
    # dask로 CSV 파일 읽기 (lazy loading)
    file_path = os.path.join(
        args['emb_file_path'], 
        f"{args['dataset_path'].split('/')[1]}_{'notlabeled' if config['is_topic_label'] == False else 'labeled'}.csv"
    )
    
    embedded_essay_raw = dd.read_csv(file_path, encoding='cp949', header=None)
    print(f"Shape: {embedded_essay_raw.shape[0].compute()} rows")
    
    # 인덱스 범위 미리 계산
    essay_ranges = []
    start_idx = 0
    for essay in essays:
        end_idx = start_idx + len(essay)
        essay_ranges.append((start_idx, end_idx))
        start_idx = end_idx
    
    # valid_index를 set으로 변환 (O(1) lookup)
    valid_set = set(valid_index)
    
    # 한 번에 필요한 데이터만 compute하여 pandas로 변환
    embedded_df = embedded_essay_raw.compute()
    
    # 벡터화된 방식으로 분할
    train_embedded_essay = [
        embedded_df.iloc[start:end] 
        for ix, (start, end) in enumerate(essay_ranges) 
        if ix not in valid_set
    ]
    
    valid_embedded_essay = [
        embedded_df.iloc[start:end] 
        for ix, (start, end) in enumerate(essay_ranges) 
        if ix in valid_set
    ]
    
    return train_embedded_essay, valid_embedded_essay

def map_to_nearest_index(arr, allowed):
    """
    arr      : 실수 배열 (…,)
    allowed  : 1‑D 배열, 매핑 대상 값들
    return   : arr 와 같은 shape, 각 위치가 allowed 의 index
    """
    diff = np.abs(arr[..., None] - allowed)   # broadcasting
    idx  = diff.argmin(axis=-1)               # 최근접 인덱스
    return idx

def mapping_pred_test(pred, test):
    value_list = np.unique(test)
    pred_idx = map_to_nearest_index(pred, value_list)
    test_idx = map_to_nearest_index(test, value_list)

    return pred_idx, test_idx

def main(args):
    
    #hyperparameters
    gc.collect()
    torch.cuda.empty_cache()
    dropout = 0.30475784381583626
    learning_rate = 0.0009154637004201508
    n_epochs = 100
    
    hidden_dim = 128
    batch_size = 128
    seed = 42

    n_models = 5
    n_outputs = len(args['rubric'])
    
    # essay 데이터셋 로드
    dataset = pd.read_csv(args['dataset_path'], encoding='utf-8-sig')
    essays , y = get_essay_dataset(is_rubric=True, args=args)

    # 인덱스 분할
    # prompt_num ="Q1"
    train_index = dataset[dataset['is_train'] == True].index
    valid_index = dataset[dataset['is_train'] == False].index

    # 벡터화된 essay 데이터셋 생성
    train_embedded_essay, valid_embedded_essay = get_embedded_essay(args, valid_index, essays)
    train_y, valid_y = y[train_index], y[valid_index]
    print(f"Train essays: {len(train_embedded_essay)}, Valid essays: {len(valid_embedded_essay)}")



    # DataLoader 설정
    train_dataset = EssayDataset(train_embedded_essay, train_y)
    valid_dataset = EssayDataset(valid_embedded_essay, valid_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # 모델, 손실 함수, 최적화기 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ensemble_predictions = []
    for model_idx in range(n_models):

        # seed 설정
        set_seed(seed + model_idx * 10)

        model = GRUScoreModule(output_dim=n_outputs,hidden_dim=hidden_dim, dropout=dropout).to(device)
        
        if model_idx > 0:
            dropout_variation = dropout + (model_idx - 2) * 0.05
            dropout_variation = max(0.2, min(0.5, dropout_variation))
            model.dropout = nn.Dropout(dropout_variation)
            
             # Ensure dropout is within [0.1, 0.5]
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 학습 및 검증 루프
        patience = 10
        train_loss_list = []
        val_loss_list = []
        best_val_loss = float('inf')
        early_stopping_counter = 0



        prev_time = time.time()
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0
            for inputs,labels in train_loader:
                inputs ,labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)        
                loss = criterion(outputs, labels)
                    
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            all_outputs = []
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.cuda(),labels.cuda()
                    outputs = model(inputs)            
                    loss = criterion(outputs, labels)
                    all_outputs.extend(outputs.cpu().numpy())
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(valid_loader)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time Elapsed: {time.time() - prev_time:.4f}')
            prev_time = time.time()
            
            if val_loss < best_val_loss:
                best_outputs = np.array(all_outputs)
                if not os.path.exists('./model'):
                    os.makedirs('./model')
                torch.save(model.state_dict(), './model/kobert_model.pth')
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping")
                    break
        ensemble_predictions.append(best_outputs)

    final_predictions = np.mean(ensemble_predictions, axis=0)

    y_pred = final_predictions * args['num_range']
    y_test = np.array(valid_y) * args['num_range']
    pred, real = mapping_pred_test(y_pred, y_test)
    metrics = compute_metrics(pred, real, args)
    print(metrics)

if __name__ == "__main__":
    args = config["nikl"]  # Change to "aihub_v1" for the other dataset
    config['is_topic_label'] = True  # Set to True if you want to include topic labels in the essays
    main(args)