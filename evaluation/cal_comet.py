# -*- coding:utf-8 _*-

import sys
from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

data = []
with open(sys.argv[1], 'r') as f1, \
    open(sys.argv[2], 'r') as f2, \
    open(sys.argv[3], 'r') as f3:
    f1 = f1.readlines()
    f2 = f2.readlines()
    f3 = f3.readlines()
    for s, o, r in zip(f1, f2, f3):
        data.append(
            {
            "src": s.strip(),
            "mt": o.strip(),
            "ref": r.strip()
            }
        )

model_output = model.predict(data, batch_size=1024, gpus=1)
print("COMET: {}".format(round(model_output[1], 4)))

