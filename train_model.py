import os
import time
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import load_data, save_history_plot

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

BASE_DIR = str(int(np.ceil(time.time())))

if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
    print(f'[INFO] {BASE_DIR} created')

X, y = load_data('data.npy')

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train.reshape(-1,))
y_test = encoder.transform(y_test.reshape(-1,))
joblib.dump(encoder, f'{BASE_DIR}\\labelencoder.pkl')

class_names = encoder.classes_

print(f'\n\t[INFO] Shape of x_train: {x_train.shape}')
print(f'\t[INFO] Shape of y_train: {y_train.shape}')
print(f'\t[INFO] Shape of x_test: {x_test.shape}')
print(f'\t[INFO] Shape of y_test: {y_test.shape}')

input_shape = (x_train.shape[1], x_train.shape[2], 1)

model = Sequential()

model.add(Conv2D(16, 2, input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(2))

model.add(Conv2D(32, 2, activation='relu'))
model.add(MaxPooling2D(2))
model.add(Dropout(0.2))

model.add(Conv2D(64, 2, activation='relu'))
model.add(MaxPooling2D(2))

model.add(Conv2D(128, 2, activation='relu'))
model.add(MaxPooling2D(2))
model.add(Dropout(0.2))

model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optmizer=Adam(lr=1e-3, decay=1e-5),
              metrics=['sparse_categorical_accuracy'])

model.summary(line_length=150)

plot_model(model, to_file=f'{BASE_DIR}\\model.png', show_shapes=True,
           dpi=200, expand_nested=True)

mc = ModelCheckpoint(f'{BASE_DIR}\\model.h5', save_best_only=True,
                     monitor='val_loss')

es = EarlyStopping(patience=30, monitor='val_loss')

history = model.fit(x_train, y_train, batch_size=32, epochs=150,
                    validation_split=0.2, callbacks=[mc, es])

model.save(f'{BASE_DIR}\\model.h5')

json_config = model.to_json(indent=4)
with open(f'{BASE_DIR}\\model_config.json', 'w') as f:
    f.write(json_config)

model.save_weights(f'{BASE_DIR}\\weights.h5')
save_history_plot(history, BASE_DIR)

print('\n\t[INFO] Saved model and training history plot')

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f'\t[INFO] Accuracy: {acc} Loss: {loss}')

y_pred = model.predict_classes(x_test)

matrix = confusion_matrix(y_test, y_pred)
matrix = matrix / matrix.astype(np.float).sum(axis=0)
df = pd.DataFrame(matrix, index=class_names, columns=class_names)

fig = plt.figure(figsize=(12, 12))
hm = sns.heatmap(df, annot=True, cmap='coolwarm')
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
plt.savefig(f'{BASE_DIR}\\confusion_matrix.png')

print('\t[INFO] Saved confusion matrix plot')

os.rename(BASE_DIR, BASE_DIR + '_loss_' + str(loss) + '__accuracy__' + str(acc))
print('\t[DONE]')
