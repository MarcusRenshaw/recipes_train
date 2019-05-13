from FeatureExtract import FeatureExtract
import os
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys # used for early exit sys.exit()
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import re

sets = FeatureExtract()

# if data files already exist, load them, or create them from raw audio files.
if (
        os.path.isfile(sets.train_X_preprocessed_data) and
        os.path.isfile(sets.train_Y_preprocessed_data) and
        os.path.isfile(sets.dev_X_preprocessed_data) and
        os.path.isfile(sets.dev_Y_preprocessed_data) and
        os.path.isfile(sets.test_X_preprocessed_data) and
        os.path.isfile(sets.test_Y_preprocessed_data)
):
    print("Preprocessed files exist, deserializing npy files")
    sets.load_deserialize_data()
else:
    print("Preprocessing raw audio files")
    sets.load_preprocess_data()

# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9, epsilon=1e-8, decay=0.
# SGD    : lr=0.01, momentum=0., decay=0.
opt = Adam()

# batch size and epochs
batch_size = 16
nb_epochs = 5

print("Training X shape: " + str(sets.train_X.shape))
print("Training Y shape: " + str(sets.train_Y.shape))
print("Dev X shape: " + str(sets.dev_X.shape))
print("Dev Y shape: " + str(sets.dev_Y.shape))
print("Test X shape: " + str(sets.test_X.shape))
print("Test Y shape: " + str(sets.test_X.shape))

input_shape = (sets.train_X.shape[1], sets.train_X.shape[2])

model = Sequential()
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, 
               return_sequences=True, input_shape=input_shape))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, 
               return_sequences=False))
model.add(Dense(units=50, activation = 'relu'))

model.add(Dense(units=sets.train_Y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer=opt, metrics=['accuracy'])
model.summary()

val_set = (sets.dev_X, sets.dev_Y)

history = model.fit(sets.train_X, sets.train_Y, batch_size=batch_size,
                    validation_data= val_set, epochs=nb_epochs)

_, train_acc = model.evaluate(sets.train_X, sets.train_Y, verbose=0)
_, dev_acc = model.evaluate(sets.dev_X, sets.dev_Y, verbose=0)
print('Train: %.3f, Dev: %.3f' % (train_acc, dev_acc))
# plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='dev')
plt.legend()
plt.show()

model_json = model.to_json()
with open("./weights/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("./weights/model_weights.h5")
print("Saved model to disk")

from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[2].output])
train_output = get_3rd_layer_output([sets.train_X])[0]

# with a Sequential model
get_3rd_layer_output_test = K.function([model.layers[0].input],
                                  [model.layers[2].output])

test_output = get_3rd_layer_output_test([sets.test_X])[0]


train_directory = './audio/train'
train_files = os.listdir(train_directory)

test_directory = './audio/test'
test_files = os.listdir(test_directory)

splitted_train_data = []
splitted_test_data = []
for test_file_data in test_files:
    for train_file_data in train_files:
        
        split_train = re.split('[ .]', train_file_data)
        train_target_label = (split_train[0]) 
        
        split_test = re.split('[ .]', test_file_data)
        test_target_label = (split_test[0])
        
        splitted_train_data.append(train_target_label)
        splitted_test_data.append(test_target_label)
        
similarity = []
for test_data in test_output:
    for train_data in train_output:
        cos_sim = cosine_similarity([train_data],[test_data])
        similarity.append(cos_sim[0][0])
      
df = pd.DataFrame({"train" : splitted_train_data, 
                   "test" : splitted_test_data, 
                   "cosine" : similarity})
df.to_csv("similarity_scores.csv", index=False)

data = pd.read_csv("similarity_scores.csv")

data = data.pivot('train', 'test' , 'cosine')

data.head()

sns.heatmap(data)

sns.clustermap(data, col_cluster=False, row_cluster=False)







