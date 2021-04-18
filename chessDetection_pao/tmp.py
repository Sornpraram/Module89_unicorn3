import numpy as np
import pandas as pd
tmp_loss = [1,2,3,4,5,6,7,8]
tmp_loss = np.array(tmp_loss)

df_summit = pd.read_csv('./train_progress.csv')
df_summit['temporory2'] = tmp_loss
df_summit.to_csv('./train_progress.csv',index=False)