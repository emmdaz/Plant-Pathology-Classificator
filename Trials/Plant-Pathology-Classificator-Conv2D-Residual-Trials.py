import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Activation, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras import regularizers, models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
import wandb

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("TensorFlow is using the GPU \n", gpus)
else:
    print("No GPU detected.")

tf.keras.backend.clear_session()

from wandb.integration.keras import WandbMetricsLogger

wandb.require("core")
wandb.login()

# Carga de los datos 

df = pd.read_csv("/Plant-Pathology-Classificator/plant-pathology-2020-/train.csv")
df.head()

import os

# convert one-hot columns to a single class name
df["label"] = df[["healthy", "multiple_diseases", "rust", "scab"]].idxmax(axis=1)

df["filepath"] = df['image_id'].apply(lambda x: os.path.join("/Plant-Pathology-Classificator/plant-pathology-2020-/images", f'{x}.jpg'))

from sklearn.model_selection import train_test_split

X_train, X_temp = train_test_split(df, test_size = 0.3, stratify = df["label"], random_state = 4)

X_test, X_val = train_test_split(X_temp, test_size = 1/3, stratify = X_temp["label"], random_state = 4)

print("Train size:", len(X_train))
print("Test size:", len(X_test))
print("Validation size:", len(X_val))

mini_train, _ = train_test_split(X_train, test_size = 0.8, stratify = X_train["label"], random_state = 4)

mini_test, _ = train_test_split(X_test, test_size = 0.75, stratify = X_test["label"], random_state = 4)

print("Small training size:", len(mini_train))
print("Small training size:", len(mini_test))

datagen = ImageDataGenerator(rescale=1./255)

train = datagen.flow_from_dataframe(
    dataframe = X_train,
    x_col = 'filepath',
    y_col = 'label',
    image_size = (128, 128),
    batch_size = 16
)

test = datagen.flow_from_dataframe(
    dataframe = X_test,
    x_col='filepath',
    y_col='label',
    image_size = (128, 128),
    batch_size = 16
)

val = datagen.flow_from_dataframe(
    dataframe = X_val,
    x_col = 'filepath',
    y_col = 'label',
    image_size = (128, 128),
    batch_size = 16
)

mini_train = train = datagen.flow_from_dataframe(
    dataframe = mini_train,
    x_col = 'filepath',
    y_col = 'label',
    image_size = (128, 128),
    batch_size = 16
)

mini_test = train = datagen.flow_from_dataframe(
    dataframe = mini_test,
    x_col = 'filepath',
    y_col = 'label',
    image_size = (128, 128),
    batch_size = 16
)

def objective(trial):

    tf.keras.backend.clear_session()

    model = models.Sequential()
    inputs = layers.Input(shape=(128, 128, 3))
    
    # Optuna sugiere número de kernels y su tamaño en la primer capa convolucional,
    # así como su función de activación
    
    kernel_1 = trial.suggest_int("Kernel_1", 10, 64)
    size_1 = trial.suggest_categorical("Kernel_Size_1", [3,7,8,10])
    activation_1 = trial.suggest_categorical("Activation_1", ["relu", "relu6", "selu", "leaky_relu", "relu6"])
    
    # Primera convolución
    x = layers.Conv2D(kernel_1, (size_1,size_1), padding = "same")(inputs)
    x = layers.Activation(activation_1)(x)
    
    # Función para los bloques residuales
    def residual_block(x, kernel, kernel_size, activation, dropout, dropout_rate, regularizer, r_1, r_2):
        
        residual = x  

        # Camino "principal"
        
        if regularizer == "L1":
            x = layers.Conv2D(kernel, (kernel_size, kernel_size), padding = "same",
                              activation = activation, kernel_regularizer = regularizers.L1(r_1))(x)
            x = layers.BatchNormalization()(x)
            
        elif regularizer == "L2":
            x = layers.Conv2D(kernel, (kernel_size, kernel_size), padding = "same",
                              activation = activation, kernel_regularizer = regularizers.L2(r_2))(x)
            x = layers.BatchNormalization()(x)
            
        elif regularizer == "L1L2":
            x = layers.Conv2D(kernel, (kernel_size, kernel_size), padding = "same",
                              activation = activation, kernel_regularizer = regularizers.L1L2(r_1,r_2))(x)
            x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(kernel, (kernel_size, kernel_size), padding = "same", activation = activation)(x)
        x = layers.BatchNormalization()(x)
        
        # Capa intermedia Dropout                      
        if dropout == "y":
            
            x = layers.Dropout(dropout_rate)(x)
        
        # Segunda capa Convolicional, lineal
        if regularizer == "L1":
            x = layers.Conv2D(kernel, (kernel_size, kernel_size), padding = "same",
                              kernel_regularizer = regularizers.L1(r_1))(x)
            x = layers.BatchNormalization()(x)
            
        elif regularizer == "L2":
            x = layers.Conv2D(kernel, (kernel_size, kernel_size), padding = "same", 
                              kernel_regularizer = regularizers.L2(r_2))(x)
            x = layers.BatchNormalization()(x)
            
        elif regularizer == "L1L2":
            x = layers.Conv2D(kernel, (kernel_size, kernel_size), padding = "same",
                              kernel_regularizer = regularizers.L1L2(r_1, r_2))(x)
            x = layers.BatchNormalization()(x)
        
        # Suma de la conexión residual
        x = layers.add([x, residual]) # Capa que suma elemento a elemento
        x = layers.Activation(activation)(x)
        return x
    
    # Optuna sugiere regularizador
    regularizer = trial.suggest_categorical("Regularizer", ["L1","L2","L1L2"])
    r_1 = trial.suggest_float("regularizer_value", 1e-6, 1e-4, log = True)
    r_2 = trial.suggest_float("regularizer_value_2", 1e-6, 1e-4, log = True)
    
    # Primer bloque residual
                              
    x = residual_block(x, kernel_1, size_1, activation_1, "n", 0.0, regularizer, r_1, r_2)
    
    kernel_per_layer = [kernel_1]
    kernel_size_per_layer = [size_1]
    activation_per_layer = [activation_1]
    
    # Optuna sugiere número de capas, número de kernels su tamaño y función de activación; también sugiere Dropout
    # y regularizadores
                              
    n_layers = trial.suggest_int("n_layers", 10, 50)
    
    dropout_per_layer = []
    dropout_percentage_per_layer = []
    
    for i in range(n_layers):
        
        kernel = trial.suggest_int(f"Kernel_{i+2}", 10, 64)
        kernel_per_layer.append(kernel)
        
        kernel_size = trial.suggest_categorical(f"Kernel_Size_{i+2}", [3,7,8,10])
        kernel_size_per_layer.append(kernel_size)
            
        activation = trial.suggest_categorical(f"Activation_{i+2}", ["relu", "relu6", "selu", "leaky_relu", "relu6"])      
        activation_per_layer.append(activation)
                              
        dropout = trial.suggest_categorical(f"Dropout_L{i+2}", ["y", "n"])
        dropout_per_layer.append(dropout)
                              
        dropout_rate = trial.suggest_float(f"Dropout_value_L{i+2}",0.2, 0.5)
        
        if dropout == "y":
            dropout_percentage_per_layer.append(dropout_rate)
        else:
            dropout_percentage_per_layer.append(0.0)
        
        # Capa Convolucional i-ésima
        if regularizer == "L1":
            x = layers.Conv2D(kernel, (kernel_size, kernel_size), strides = 2, padding = "same", activation = activation,
                          kernel_regularizer = regularizers.L1(r_1))(x)
        elif regularizer == "L2":
            x = layers.Conv2D(kernel, (kernel_size, kernel_size), strides = 2, padding = "same", activation = activation,
                          kernel_regularizer = regularizers.L2(r_2))(x)
        else: 
            x = layers.Conv2D(kernel, (kernel_size, kernel_size), strides = 2, padding = "same", activation = activation,
                          kernel_regularizer = regularizers.L1L2(r_1,r_2))(x)
    
        x = layers.BatchNormalization()(x)
        
        # Bloque residual i-ésimo
        x = residual_block(x, kernel, kernel_size, activation, dropout, dropout_rate, regularizer, r_1, r_2)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
                              
    outputs = layers.Dense(4, activation = "softmax")(x)

    # Optuna sugiere Learning Rate y Optimizador
    
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
                              
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate = lr)
                              
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate = lr)
        
        
    model = models.Model(inputs, outputs)
                              
    model.compile(
    optimizer = optimizer,
    loss = "categorical_crossentropy",
        metrics = ["accuracy"])

    wandb.init(
        project = "Plant-Pathology-Classificator-Conv2D-Residual-Trials",
        name = f"Trial_{trial.number}",
        reinit = True,
        config = {
            "n_layers": n_layers,
            "kernel_per_layer": kernel_per_layer,
            "kernel_size_per_layer": kernel_size_per_layer,
            "activations_per_layer": activation_per_layer,
            "regularizer": regularizer,
            "r_value": r_1,
            "r_value2": r_2,
            "dropout_per_layer": dropout_per_layer,
            "dropout_percentage_per_layer": dropout_percentage_per_layer,
            "learning_rate": lr,
            "optimizer": optimizer_name
        }
    )
                              
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 8, restore_best_weights = True)
    
    try:
        history = model.fit(
            mini_train, validation_data = mini_test,
            epochs = 100,
            batch_size = 16,
            verbose = 0, 
            callbacks = [WandbMetricsLogger(log_freq=5), early_stopping]
        )

        val_loss = min(history.history["val_loss"])
        train_loss = min(history.history["loss"])
    
    except tf.errors.ResourceExhaustedError:
        tf.keras.backend.clear_session()
        wandb.finish()
        return float("inf")

    # Penalize overfitting
    score = val_loss + 0.1 * (train_loss - val_loss)
    
    tf.keras.backend.clear_session()
    gc.collect()
    wandb.finish()

    return score

study = optuna.create_study(study_name = "Proyecto", direction = "minimize")
study.optimize(objective, n_trials = 60)

print("Número de pruebas terminadas: ", len(study.trials))

trial = study.best_trial

print("Mejor intento: ", trial)

print("Valor: ", trial.value)
print("Hiperparámetros: ", trial.params)

from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_slice
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank

import plotly.io as pio

pio.renderers.default = "notebook_connected"

plot_parallel_coordinate(study)

plot_param_importances(study)