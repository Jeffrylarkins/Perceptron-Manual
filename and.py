from utils.model import perceptron
from utils.all_utils import prepare_data, save_model
import pandas as pd
import numpy as np
import logging

logging_str = "[%(asctime)s: %(levelname)s:%(module)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_str)
AND = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1]
}

df = pd.DataFrame(AND)
logging.info(df)
X,y = prepare_data(df)

ETA = 0.3
EPOCHS = 100
model = perceptron(eta = ETA, epochs = EPOCHS) 
model.fit(X,y)

_ = model.total_loss()


save_model(model,'and.model')