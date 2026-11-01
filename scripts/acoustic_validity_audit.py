"""Development-only metadata and audio-summary confounding audit; never reads test audio."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/real-training-cuda/full/treevibes-984e68c4e99668c8'
OUT = ROOT / 'outputs/acoustic-validity-audit'
OUT.mkdir(parents=True, exist_ok=True)
m = pd.read_csv(RUN / 'manifest.csv').merge(pd.read_csv(RUN / 'splits.csv')[['path', 'split']], on='path', validate='one_to_one')
dev = m[m.split.isin(['train', 'val'])].copy()
train, val = (dev[dev.split == s].copy() for s in ('train', 'val'))
def features(frame):
    return np.column_stack([np.log10(frame.rms.to_numpy().clip(1e-12)), frame.duration_seconds, frame.clipping_fraction])

def metrics(y, predicted):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    def div(a, b): return float(a/b) if b else None
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), recall=div(tp,tp+fn), fpr=div(fp,fp+tn), precision=div(tp,tp+fp), f1=div(2*tp,2*tp+fp+fn))

# Train-only features and threshold, no validation tuning. 95th percentile FPR is approximate under ties.
audio = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
audio.fit(features(train), train.label)
train_s = audio.predict_proba(features(train))[:, 1]
val_s = audio.predict_proba(features(val))[:, 1]
threshold = float(np.quantile(train_s[train.label.to_numpy() == 0], .95, method='higher'))
# Known component IDs cannot generalize to new components: default to train majority.
source_mapping = train.groupby('component').label.agg(lambda x: int(x.mean() >= .5)).to_dict()
default = int(train.label.mean() >= .5)
source_pred = val.component.map(source_mapping).fillna(default).astype(int).to_numpy()
# Deliberately within-training-recording split: a shortcut diagnostic, NOT a held-out source result.
a,b=train_test_split(np.arange(len(train)),test_size=.3,random_state=0,stratify=train.component)
source_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,random_state=0))
source_model.fit(features(train)[a],train.component.iloc[a])
source_accuracy = accuracy_score(train.component.iloc[b],source_model.predict(features(train)[b]))
majority_accuracy = train.component.iloc[b].value_counts(normalize=True).max()
print('source_component_accuracy', source_accuracy, 'majority', majority_accuracy)
results={'source_only_unseen_components_fallback':metrics(val.label,source_pred),
         'simple_audio_train_threshold':metrics(val.label,val_s>=threshold),
         'simple_audio_threshold':threshold,
         'source_identity_within_train_recordings_accuracy':float(source_accuracy),
         'source_identity_within_train_recordings_majority_accuracy':float(majority_accuracy),
         'source_identity_warning':'Random recording split within same components; descriptive shortcut probe only, not independent generalization.',
         'label_by_train_component':pd.crosstab(train.component,train.label).to_dict(orient='index')}
blend=pd.read_csv(ROOT/'outputs/model-improvement/best_blend_validation_predictions.csv')
val=val.merge(blend[['path','label','cnn_score','blend_score']],on=['path','label'],validate='one_to_one')
threshold_cnn=json.loads((RUN/'selection_draft.json').read_text())['selected'][0]['threshold']
threshold_blend=json.loads((ROOT/'outputs/model-improvement/best_blend_exploratory.json').read_text())['threshold']
results['cnn_validation_selected']=metrics(val.label,val.cnn_score.to_numpy(dtype=np.float32)>=np.float32(threshold_cnn))
results['blend_validation_selected']=metrics(val.label,val.blend_score>=threshold_blend)
val['simple_prediction']=(val_s>=threshold).astype(int)
val['cnn_prediction']=(val.cnn_score.to_numpy(dtype=np.float32)>=np.float32(threshold_cnn)).astype(int)
val['blend_prediction']=(val.blend_score>=threshold_blend).astype(int)
rows=[]
for group,frame in val.groupby('group_id'):
    for model in ('simple','cnn','blend'):
        rows.append(dict(folder=group,model=model,n=len(frame),positive_labels=int(frame.label.sum()),**metrics(frame.label,frame[model+'_prediction'])))
pd.DataFrame(rows).to_csv(OUT/'validation_by_folder.csv',index=False)
pd.crosstab([dev.split,dev.component,dev.group_id],dev.label).to_csv(OUT/'development_components.csv')
(OUT/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf8')
print(json.dumps(results,indent=2))
