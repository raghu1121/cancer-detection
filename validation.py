from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report


seed = 7
np.random.seed(seed)
df=pd.read_csv('dataset.csv')
X = df.loc[:,:'Ancestry_WHITE']
Y = df.loc[:,'Cancer type_Daisy':'Cancer type_control']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean=False)
X.loc[:,'Age':'Max'] = scaler.fit_transform(X.loc[:,'Age':'Max'])
X = np.expand_dims(X, axis=2)




model = load_model('models-0.9643.hdf5')
Y_pred = model.predict(X)
predicted = np.argmax(Y_pred, axis=1)
true = np.argmax(Y.to_numpy(), axis=1)

report = classification_report(true, predicted)
print(report)