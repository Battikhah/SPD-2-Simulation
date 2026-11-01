"""Frozen CNN sensitivity to synthetic broadband noise on a fixed validation subset."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/real-training-cuda/full/treevibes-984e68c4e99668c8'
TRIAL = RUN / 'trials/0f76c3e188ea66a4b59ac274'


class LogMelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                                  nn.Flatten(), nn.Linear(32, 1))

    def forward(self, x):
        return self.body(x[:, None]).squeeze(-1)


def main():
    manifest = pd.read_csv(RUN/'manifest.csv').merge(pd.read_csv(RUN/'splits.csv')[['path', 'split']], on='path', validate='one_to_one')
    rng = np.random.default_rng(19)
    selected = manifest[manifest.split == 'val'].groupby('group_id', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 5), random_state=19), include_groups=False)
    ids = set(selected.index)
    cache = None
    for path in RUN.glob('features-*.npz'):
        with np.load(path) as z:
            record_ids = z['record_id']
            if set(record_ids).issubset(set(manifest.index[manifest.split == 'val'])):
                cache = (record_ids, z['raw'])
                break
    if cache is None:
        raise RuntimeError('No validation-only feature cache found')
    record_ids, raw = cache
    mask = np.isin(record_ids, list(ids))
    record_ids, raw = record_ids[mask], raw[mask]
    assert set(record_ids) == ids and set(manifest.iloc[record_ids].split) == {'val'}
    bundle = joblib.load(TRIAL/'model.joblib')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = LogMelCNN().to(device)
    net.load_state_dict(bundle['state'])
    net.eval()
    freq = np.arange(257)*8000/512
    hz = 700*(10**(np.linspace(0, 2595*np.log10(1+4000/700), 66)/2595)-1)
    mel = np.maximum(0, np.minimum((freq[None]-hz[:-2,None])/(hz[1:-1,None]-hz[:-2,None]), (hz[2:,None]-freq[None])/(hz[2:,None]-hz[1:-1,None])))
    hann = np.hanning(513)[:-1]
    def scores(audio):
        result = []
        with torch.no_grad():
            for start in range(0, len(audio), 128):
                batch = audio[start:start+128]
                frames = np.stack([batch[:, j*128:j*128+512] for j in range(59)], axis=1)
                power = np.abs(np.fft.rfft(frames*hann, axis=2))**2/512
                features = np.log(np.maximum(power @ mel.T, 1e-10)).astype('float32')
                tensor = torch.as_tensor((features-bundle['mean'])/bundle['std'], device=device)
                result.extend(torch.sigmoid(net(tensor)).cpu().numpy())
        return pd.DataFrame({'record_id':record_ids,'score':result}).groupby('record_id').score.mean()
    clean = scores(raw)
    source_rms = np.sqrt(np.mean(raw.astype('float64')**2, axis=1, keepdims=True))
    noise = rng.normal(size=raw.shape)
    noise = noise / np.maximum(np.sqrt(np.mean(noise**2, axis=1, keepdims=True)), 1e-12) * source_rms * 10**(-10/20)
    perturbed = scores(raw + noise)
    saved = pd.read_csv(TRIAL/'validation_predictions.csv').set_index('record_id').score
    assert np.max(np.abs(clean - saved.loc[clean.index])) < 2e-5, 'Clean inference does not reproduce frozen model scores'
    y = manifest.loc[clean.index, 'label'].to_numpy()
    threshold = np.float32(bundle['threshold'])
    clean_pred, noisy_pred = clean.to_numpy() >= threshold, perturbed.to_numpy() >= threshold
    print('device',device,'recordings',len(clean),'changed predictions',int(np.sum(clean_pred != noisy_pred)))
    for name, pred in [('clean',clean_pred), ('noise_10db',noisy_pred)]:
        print(name, 'tn fp fn tp', confusion_matrix(y,pred,labels=[0,1]).ravel().tolist())
    output = pd.DataFrame({'record_id':clean.index, 'label':y, 'clean_score':clean.to_numpy(), 'noise_score':perturbed.to_numpy(), 'clean_prediction':clean_pred, 'noise_prediction':noisy_pred})
    output.to_csv(ROOT/'outputs/acoustic-validity-audit/noise_sensitivity.csv', index=False)


if __name__ == '__main__':
    main()
