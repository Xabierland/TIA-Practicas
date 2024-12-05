# === Librerías ===

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing import text
from sklearn.utils import shuffle
import re
import pandas as pd
import matplotlib.pyplot as plt
import time

# === Funciones ===

def load_data(path):
    training_set = load_sst_data(path+'train.txt')
    dev_set = load_sst_data(path+'dev.txt')
    test_set = load_sst_data(path+'test.txt')
    return training_set, dev_set, test_set

def load_sst_data(path,
                  easy_label_map={0:0, 1:0, 2:None, 3:1, 4:1}):
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            example = {}
            example['label'] = easy_label_map[int(line[1])]
            if example['label'] is None:
                continue

            # Strip out the parse information and the phrase labels---we don't need those here
            text = re.sub(r'\s*(\(\d)|(\))\s*', '', line)
            example['text'] = text[1:]
            data.append(example)
    data = pd.DataFrame(data)
    return data

def preprocess_data(training_set, dev_set, test_set):
    # Shuffle dataset
    training_set = shuffle(training_set)
    dev_set = shuffle(dev_set)

    test_set = shuffle(test_set)

    # Obtain text and label vectors, and tokenize the text
    train_texts = training_set.text
    train_labels = training_set.label

    dev_texts = dev_set.text
    dev_labels = dev_set.label

    test_texts = test_set.text
    test_labels = test_set.label

    # Create a tokenize that takes the 1000 most common words
    tokenizer = text.Tokenizer(num_words=1000)

    # Build the word index (dictionary)
    tokenizer.fit_on_texts(train_texts) # Create word index using only training part

    # Vectorize texts into one-hot encoding representations
    x_train = tokenizer.texts_to_matrix(train_texts, mode='binary')
    x_dev = tokenizer.texts_to_matrix(dev_texts, mode='binary')
    x_test = tokenizer.texts_to_matrix(test_texts, mode='binary')

    y_train = train_labels
    y_dev = dev_labels
    y_test = test_labels
    
    return x_train, y_train, x_dev, y_dev, x_test, y_test

def create_model(input_shape, hidden_units, dropout_rate, l1_lambda, l2_lambda):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_shape,)))
    for units in hidden_units:
        model.add(tf.keras.layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda)))
        model.add(Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

def train_model(model, x_train, y_train, x_dev, y_dev, epochs, batch_size):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_dev, y_dev), callbacks=[early_stopping])
    return model, history

def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy

def run_experiment(hidden_units, dropout_rate, l1_lambda, l2_lambda, epochs, batch_size):
    model = create_model(x_train.shape[1], hidden_units, dropout_rate, l1_lambda, l2_lambda)
    model, history = train_model(model, x_train, y_train, x_dev, y_dev, epochs, batch_size)
    loss, accuracy = evaluate_model(model, x_test, y_test)
    return loss, accuracy, history

def draw_results(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Dev'], loc='upper left')
    
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Dev'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

# === Main ===

if __name__ == "__main__":
    # Iniciamos una seed tanto para numpy como para tensorflow
    np.random.seed(1)
    tf.random.set_seed(2)
    
    # Cargamos los datos
    training_set, dev_set, test_set = load_data('Data/')
    # Preprocesamos los datos
    x_train, y_train, x_dev, y_dev, x_test, y_test = preprocess_data(training_set, dev_set, test_set)
    
    # Definimos los hiperparámetros
    hidden_units_list = [[50, 50], [100, 50], [100, 100]]
    dropout_rate_list = [0.3, 0.5, 0.7]
    l1_lambda_list = [0.001, 0.01, 0.1]
    l2_lambda_list = [0.001, 0.01, 0.1]
    batch_size_list = [32, 64, 128]
    epochs = 100

    best_accuracy = 0
    best_params = None
    best_history = None

    startTime = time.time()
    for hidden_units in hidden_units_list:
        for dropout_rate in dropout_rate_list:
            for l1_lambda in l1_lambda_list:
                for l2_lambda in l2_lambda_list:
                    for batch_size in batch_size_list:
                        print(f"Training with hidden_units={hidden_units}, dropout_rate={dropout_rate}, l1_lambda={l1_lambda}, l2_lambda={l2_lambda}, batch_size={batch_size}")
                        loss, accuracy, history = run_experiment(hidden_units, dropout_rate, l1_lambda, l2_lambda, epochs, batch_size)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_loss = loss
                            best_params = (hidden_units, dropout_rate, l1_lambda, l2_lambda, batch_size)
                            best_history = history
    print(f"Training time: {time.time()-startTime}")

    print(f"Best accuracy & lost: {best_accuracy} & {best_loss} with parameters: hidden_units={best_params[0]}, dropout_rate={best_params[1]}, l1_lambda={best_params[2]} l2_lambda={best_params[3]}, batch_size={best_params[4]}")
    
    # Dibujamos los resultados del mejor modelo
    draw_results(best_history)
    