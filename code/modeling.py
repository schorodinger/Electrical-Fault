import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras import optimizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#df = pd.read_csv('data\processed\df_cut.csv')
df = pd.read_csv('data\processed\df_cut_unif.csv', converters={'faultCode': lambda x: str(x)})

df.shape

x, y = df[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']], df['faultCode']
y = pd.get_dummies(y)
y.rename(columns={'0000': 'NF', '0110': 'LL', '0111': 'LLL', '1001': 'LG', '1011': 'LLG'}, inplace=True)  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)


# scaler = StandardScaler()
scaler = MinMaxScaler()
x_train_scl = scaler.fit_transform(x_train)
x_test_scl = scaler.fit_transform(x_test)


n_features = x.shape[1]
no_classes = len(y.columns.unique())
recompiling_optimizer = optimizers.Adam(learning_rate=0.001)


model = Sequential()
model.add(Dense(n_features, activation='relu', input_shape=(n_features, )))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

model.compile(loss="categorical_crossentropy",
            optimizer=recompiling_optimizer,
            metrics=['accuracy'])

history = model.fit(x_train_scl, y_train,
        epochs=3000,
        validation_data=(x_test_scl, y_test),
        verbose=1)

scores = model.evaluate(x_test_scl, y_test, verbose=0)
print(f'Score: {model.metrics_names[0]} de {scores[0]}; {model.metrics_names[1]} de {scores[1]*100}%')


sns.set(rc={'axes.facecolor':'efeae5', 'figure.facecolor':'efeae5'})
fig, ax = plt.subplots(1, 2, figsize=(24, 8))
sns.lineplot(data=history.history['accuracy'], ax=ax[0], color='blue')
sns.lineplot(data=history.history['val_accuracy'], ax=ax[0], color='#f77f00')
ax[0].set_title('Acuracia por epoca', {'fontsize': 16})
ax[0].set_xlabel('épocas')
ax[0].set_ylabel('acurácia')
plt.legend(['treino', 'validacao'])
sns.lineplot(data=history.history['loss'], ax=ax[1], color='blue')
sns.lineplot(data=history.history['val_loss'], ax=ax[1], color='#f77f00')
ax[1].set_title('Perda por epoca', {'fontsize': 16})
ax[1].set_xlabel('épocas')
ax[1].set_ylabel('perda')
plt.legend(['treino', 'validacao'])
plt.show()


y_pred = model.predict(x_test_scl)
y_hat_classes = np.argmax(y_pred, axis=1)
y_real_classes =  np.argmax(y_test.values, axis=1)
cmf = confusion_matrix(y_real_classes, y_hat_classes)

sns.set(rc={'axes.facecolor':'efeae5', 'figure.facecolor':'efeae5'})
plt.figure(figsize=(12,12))
sns.heatmap(cmf, annot=True, fmt='g', cmap='Reds', linecolor='gray')
plt.xlabel('P R E V I S T O')
plt.ylabel('R E A L')
plt.title('Confusion Matrix')
plt.show()