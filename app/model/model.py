from fastai.vision.all import *
from pathlib import Path

#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/coral_learn-{__version__}.pkl", "rb") as f:
    learn = load_learner(f)

labels = learn.dls.vocab

def learn_predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}