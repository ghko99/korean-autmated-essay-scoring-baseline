# ================================================================
#  Embedding‑기반 에세이 채점 모델 (GRU + LayerNorm + Dropout)
#  ─ 수정 내용: heavy I/O(데이터셋·임베딩) 단 1회만 실행
# ================================================================
import os, time, json, gc, random
import numpy as np
import pandas as pd
import dask.dataframe as dd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score
from psutil import virtual_memory   # 사용 여부는 선택
from embedding import get_essay_dataset
from config import config

# ----------------------------- 모델 -----------------------------
class GRUScoreModule(nn.Module):
    def __init__(self, output_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.gru(x)          # [B, T, 2H]
        x = x[:, -1, :]             # 마지막 타임스텝
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x)


# ------------------------- Dataset 클래스 ------------------------
class EssayDataset(Dataset):
    def __init__(self, embedded_essays, labels, maxlen=128):
        self.embedded_essays = embedded_essays
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.maxlen = maxlen

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        essay = self.embedded_essays[idx]
        if hasattr(essay, "values"):          # DataFrame → np.array
            essay = essay.values
        if len(essay) < self.maxlen:          # pre‑padding
            padded = np.zeros((self.maxlen, essay.shape[1]), dtype=np.float32)
            padded[-len(essay) :] = essay
        else:                                 # truncate
            padded = essay[: self.maxlen].astype(np.float32)
        return torch.tensor(padded), self.labels[idx]


# ------------------------ 유틸리티 함수 -------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_pred_idx, y_true_idx, rubric):
    metrics = {}
    all_kappa = []
    for i, rub in enumerate(rubric):
        yp, yt = y_pred_idx[:, i], y_true_idx[:, i]
        metrics[rub] = {
            "accuracy": accuracy_score(yt, yp),
            "kappa": cohen_kappa_score(yt, yp, weights="quadratic"),
        }
        all_kappa.append(metrics[rub]["kappa"])

    metrics["mean"] = {
        "accuracy": accuracy_score(y_true_idx.flatten(), y_pred_idx.flatten()),
        "kappa": np.mean(all_kappa),
    }
    metrics["overall"] = {
        "kappa": cohen_kappa_score(
            y_true_idx.flatten(), y_pred_idx.flatten(), weights="quadratic"
        )
    }
    return metrics


def map_to_nearest_index(arr, allowed):
    diff = np.abs(arr[..., None] - allowed)  # broadcasting
    return diff.argmin(axis=-1)              # recent index


def mapping_pred_test(pred, test):
    allowed_vals = np.unique(test)
    return map_to_nearest_index(pred, allowed_vals), map_to_nearest_index(
        test, allowed_vals
    )


# ------------------- 임베딩 분할 (I/O 제거 버전) -------------------
def get_embedded_essay(valid_index, essays, embedded_df):
    essay_ranges, start = [], 0
    for essay in essays:
        end = start + len(essay)
        essay_ranges.append((start, end))
        start = end
    valid_set = set(valid_index)

    train_embs = [
        embedded_df.iloc[s:e] for ix, (s, e) in enumerate(essay_ranges) if ix not in valid_set
    ]
    valid_embs = [
        embedded_df.iloc[s:e] for ix, (s, e) in enumerate(essay_ranges) if ix in valid_set
    ]
    return train_embs, valid_embs


# ---------------------------- Main ------------------------------
def main(args, prompt_groups, dataset, essays, y, embedded_df):
    # 하이퍼파라미터
    dropout = 0.30475784381583626
    lr = 9.154637004201508e-4
    n_epochs = 100
    hidden_dim = 128
    batch_size = 128
    seed = 42
    patience = 10
    n_outputs = len(args["rubric"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    # ---------- Train / Valid Split ----------
    ## Cross-Prompt 실험을 위한 세팅
    if config["mode"].startswith("group_"):
        valid_prompts = prompt_groups[config["mode"]]
        train_idx = dataset[~dataset["essay_main_subject"].isin(valid_prompts)].index
        valid_idx = dataset[dataset["essay_main_subject"].isin(valid_prompts)].index
    else:
        train_idx = dataset[dataset["is_train"]].index
        valid_idx = dataset[~dataset["is_train"]].index

    # ---------- 임베딩 / 라벨 분할 ----------
    train_embs, valid_embs = get_embedded_essay(valid_idx, essays, embedded_df)
    y_train, y_valid = y[train_idx], y[valid_idx]

    # ---------- DataLoader ----------
    train_loader = DataLoader(
        EssayDataset(train_embs, y_train), batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        EssayDataset(valid_embs, y_valid), batch_size=batch_size, shuffle=False
    )

    # ---------- 모델 학습 ----------
    model = GRUScoreModule(n_outputs, hidden_dim, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_outputs = None
    early_stop = 0
    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, outputs = 0.0, []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item()
                outputs.append(out.cpu().numpy())
        val_loss /= len(valid_loader)
        train_loss /= len(train_loader)

        print(
            f"[{epoch:03d}/{n_epochs}] "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"Δt {time.time() - start_time:.1f}s"
        )
        start_time = time.time()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_outputs = np.vstack(outputs)
            torch.save(model.state_dict(), "./model/kobert_model.pth")
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= patience:
                print("Early stopping.")
                break

    # ---------- 평가 ----------
    y_pred = best_outputs * args["num_range"]
    y_true = np.array(y_valid) * args["num_range"]
    pred_idx, true_idx = mapping_pred_test(y_pred, y_true)
    metrics = compute_metrics(pred_idx, true_idx, args["rubric"])

    # ---------- 결과 저장 ----------
    os.makedirs(config["result_path"], exist_ok=True)
    with open(
        os.path.join(config["result_path"], f"{'not_topic_' if not config['is_topic_label'] else ''}{config['mode']}_metrics.txt"), "w"
    ) as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    return metrics["mean"]["kappa"], metrics["overall"]["kappa"]


# ----------------------- 진입점 -----------------------
if __name__ == "__main__":
    # 0) 공통 설정
    args = config["aihub_v1"]        # 필요 시 다른 데이터셋으로 교체
    config["is_topic_label"] = False  # 토픽 라벨 포함 여부

    # 1) heavy I/O (딱 한 번)
    DATASET = pd.read_csv(args["dataset_path"], encoding="utf-8-sig")
    ESSAYS, Y = get_essay_dataset(is_rubric=True, args=args)

    emb_file = os.path.join(
        args["emb_file_path"],
        f"{args['dataset_path'].split('/')[1]}_{'notlabeled' if not config['is_topic_label'] else 'labeled'}.csv",
    )
    EMBEDDED_DF = dd.read_csv(emb_file, encoding="cp949", header=None).compute()

    mean_kappa , overall_kappa = main(
        args, None, DATASET, ESSAYS, Y, EMBEDDED_DF
    )
    print(f"Mean Kappa: {mean_kappa}, Overall Kappa: {overall_kappa}")
