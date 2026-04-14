import torch, yaml
from src.models.fault_classifier import FaultClassifier
from src.data.task_splitter import TaskSplitter
from collections import defaultdict

cfg = yaml.safe_load(open("config/config.yaml"))
model = FaultClassifier.from_config("config/config.yaml")
model.load_state_dict(torch.load("results/static_task0.pth"))
model = model.cuda().eval()

ts = TaskSplitter(cfg)
_, tl = ts.get_task(0)
pc = defaultdict(lambda: {"c": 0, "t": 0})
ci = torch.tensor([0, 1, 2, 3], device="cuda")

with torch.no_grad():
    for x, y in tl:
        x, y = x.cuda().float(), y.cuda().long()
        pred = ci[model(x)[:, ci].argmax(1)]
        for i in range(y.size(0)):
            pc[y[i].item()]["t"] += 1
            pc[y[i].item()]["c"] += (pred[i] == y[i]).item()

for c in sorted(pc):
    print(f"Class {c}: {pc[c]['c']}/{pc[c]['t']} = {pc[c]['c']/pc[c]['t']*100:.1f}%")
