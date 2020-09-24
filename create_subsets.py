import joblib
import json
import numpy as np

passage_ids_path = "data/passage_ids.pkl"
passage_reps_path = "data/passage_reps.pkl"
qrel_path = "data/qrels.txt"
Q_rel_count = 3000

new_qrel_out = "data/new_qrels.txt"
new_pids_out = "data/new_pids.pkl"
new_preps_out = "data/new_preps.pkl"

print("loading ids")
with open(passage_ids_path, 'rb') as handle:
    passage_ids = joblib.load(handle)

print("loading passages")
with open(passage_reps_path, 'rb') as handle:
    passage_reps = joblib.load(handle)

print("loading qrels")
with open(qrel_path) as handle:
    qrels = json.load(handle)

passage_id_to_idx = {}
for i, pid in enumerate(passage_ids):
    passage_id_to_idx[pid] = i

print("now splitting")

new_pids = []
new_preps = []
new_qrels = {}

for i, (qid, v) in enumerate(qrels.items()):
    if i < Q_rel_count:
        new_qrels[qid] = v
        for pid in v.keys():
            new_pids.append(pid)
            new_preps.append(passage_reps[passage_id_to_idx[pid]])
    else:
        break

print("all done, now saving")

with open(new_qrel_out, "w") as jsondumpfile:
    json.dump(new_qrels, jsondumpfile)

joblib.dump(np.array(new_pids), new_pids_out)
joblib.dump(np.array(new_preps), new_preps_out)




