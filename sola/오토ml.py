import pandas as pd
from pycaret.classification import *
import pandas as pd

data_path = '/Users/sola/Downloads/open/'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')
train_x = train.drop(['TIMESTAMP','PRODUCT_ID'], axis=1)
train_y = train['Y_Class']
train_x = train_x.fillna(0)

clf = setup(data = train_x, target = train_y, train_size = 0.7,  use_gpu = True, data_split_shuffle=True, normalize = True, session_id=42)



models()


best_5 = compare_models(sort = 'Accuracy', n_select = 5)