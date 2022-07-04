import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard import errors

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras import optimizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score


def build_clf(unit):
    model = Sequential()
    
    model.add(Dense(n_features, activation='relu', input_shape=(n_features, )))
    model.add(Dense(unit, activation='relu'))
    model.add(Dense(unit, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))
    
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
    
    return model

# Caregando o dataset
df = pd.read_csv('C:/Users/LUCAS/electrical_fault/data/processed/df_cut_unif.csv', converters={'faultCode': lambda x: str(x)})

# Dividindo o dataset em features/labels e treino/teste
x, y = df[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']], df['faultCode']
# y = pd.get_dummies(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

no_classes = len(y.unique()) # n° de classes para output do modelo
n_features = x.shape[1] # n° de features para input do modelo

# TENTATIVAS
# y_train_cat =  to_categorical(y_train, num_classes=no_classes)
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

# Escalando os dados
scaler = MinMaxScaler()
x_train_scl = scaler.fit_transform(x_train)
x_test_scl = scaler.fit_transform(x_test)




# Criando o modelo com KFold e GridSearch
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

clf = KerasClassifier(build_fn=build_clf, epochs=2000)

params = {'unit': [6, 8, 12, 16, 20]}

grid = GridSearchCV(estimator=clf, param_grid=params, cv=kfold, scoring='accuracy', verbose=3)

#y_train = pd.get_dummies(y_train)
#y_test = pd.get_dummies(y_test)


history = grid.fit(x_train, y_train)







fig, ax = plt.subplots(1, 2, figsize=(16, 5))
sns.lineplot(data=history.history['accuracy'], ax=ax[0])
sns.lineplot(data=history.history['val_accuracy'], ax=ax[0])
ax[0].set_title('Acuracia por epoca')
ax[0].set_xlabel('épocas')
ax[0].set_ylabel('acurácia')
plt.legend(['treino', 'validacao'])
sns.lineplot(data=history.history['loss'], ax=ax[1])
sns.lineplot(data=history.history['val_loss'], ax=ax[1])
ax[1].set_title('Perda por epoca')
ax[1].set_xlabel('épocas')
ax[1].set_ylabel('perda')
plt.legend(['treino', 'validacao'])

plt.show()       