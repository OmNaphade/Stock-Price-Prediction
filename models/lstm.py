"""A small sliding-window LSTM behind the exact same Predictor interface
as the tabular models (`fit(X, y)` / `predict(X)`). The backtester and
service layer don't know or need to know this one is a neural net.

The one real wrinkle: `predict(X)` on the Predictor Protocol assumes each
row maps to one prediction, but an LSTM needs `lookback` rows of *history*
before it can predict the first one. This class resolves that by carrying
the tail of whatever it was last `fit()` on as context for the start of
the next `predict()` call — legitimate, since in both the walk-forward
backtester and the live-prediction path, `predict()` is always called on
the fold/window immediately following the one just fit on, so that tail is
real past data, not a peek at the future.

Meaningfully slower to fit than the tabular models (walk-forward
backtesting re-trains it from scratch per fold) — a deliberately chosen
model, not a default. Only registered as an available model at all when
PyTorch is importable; see `services/prediction_service.py`.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class _LSTMNet(nn.Module):
        def __init__(self, n_features: int, hidden_size: int):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return self.head(h_n[-1])


class LSTMReturnPredictor:
    name = "lstm"

    def __init__(self, lookback: int = 20, hidden_size: int = 16, epochs: int = 30, lr: float = 1e-3):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed; LSTMReturnPredictor is unavailable.")
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self._model: "_LSTMNet | None" = None
        self._context: np.ndarray | None = None

    def _make_windows(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(X)
        if n < self.lookback:
            return np.empty((0, self.lookback, X.shape[1])), np.empty((0,))
        sequences = np.stack(
            [X[i - self.lookback + 1 : i + 1] for i in range(self.lookback - 1, n)]
        )
        targets = y[self.lookback - 1 :]
        return sequences.astype(np.float32), targets.astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        torch.manual_seed(42)
        n_features = X.shape[1]
        self._model = _LSTMNet(n_features, self.hidden_size)

        sequences, targets = self._make_windows(X, y)
        if len(sequences) > 0:
            seq_tensor = torch.from_numpy(sequences)
            target_tensor = torch.from_numpy(targets).unsqueeze(-1)
            optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()
            self._model.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                loss = loss_fn(self._model(seq_tensor), target_tensor)
                loss.backward()
                optimizer.step()

        tail = max(0, self.lookback - 1)
        self._context = X[-tail:] if tail else np.empty((0, n_features))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or len(X) == 0:
            return np.zeros(len(X), dtype=np.float64)

        context = self._context if self._context is not None else np.empty((0, X.shape[1]))
        full = np.concatenate([context, X], axis=0)

        windows = []
        for i in range(len(context), len(full)):
            window = full[max(0, i - self.lookback + 1) : i + 1]
            if len(window) < self.lookback:
                pad = np.repeat(window[:1], self.lookback - len(window), axis=0)
                window = np.concatenate([pad, window], axis=0)
            windows.append(window)

        seq_tensor = torch.tensor(np.stack(windows), dtype=torch.float32)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(seq_tensor).squeeze(-1).numpy()
        return preds.astype(np.float64)
