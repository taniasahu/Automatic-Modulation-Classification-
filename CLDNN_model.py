import numpy as np
import scipy.io as sio
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, AveragePooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# Load dataset
data = sio.loadmat(r"C:\Users\tania\Downloads\AMC_dataset.mat")
X = data['data']  # Shape: (55000, 1024, 2) for I/Q data
y = data['labels'].flatten() - 1  # Convert to 0-based indexing (0-10 for 11 classes)
split = data['split'].flatten()

# Split dataset based on the split array (70/15/15)
X_train = X[split == 0]
y_train = y[split == 0]
X_val = X[split == 1]
y_val = y[split == 1]
X_test = X[split == 2]
y_test = y[split == 2]

# Small data augmentation (Gaussian Noise)
noise = np.random.normal(0, 0.01, X_train.shape)
X_train = X_train + noise

# Define the CLDNN model
# Input Layer
input_layer = Input(shape=(1024, 2))  # 1024-sample frames with I/Q channels

# Convolutional blocks (6x [bn, relu, conv, bn, relu, conv, avgpool])
conv_out = input_layer
for _ in range(6):
    conv1 = Conv1D(64, kernel_size=3, padding='same')(conv_out if _ == 0 else conv2)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)
    conv2 = Conv1D(64, kernel_size=3, padding='same')(relu1)
    bn2 = BatchNormalization()(conv2)
    relu2 = ReLU()(bn2)
    conv_out = AveragePooling1D(pool_size=2)(relu2)

# LSTM layer for temporal dependencies
lstm_out = LSTM(256, return_sequences=False)(conv_out)  

# Fully connected layers with L2 regularization
flatten = Flatten()(lstm_out)
dense1 = Dense(512, activation='relu', kernel_regularizer=l2(0.03))(flatten)  # L2 at 0.03
bn_dense = BatchNormalization()(dense1)  # Batch normalization
dropout = Dropout(0.6)(bn_dense)  # dropout at 0.6
output = Dense(11, activation='softmax', kernel_regularizer=l2(0.03))(dropout)  

# Create model
model = Model(inputs=input_layer, outputs=output)

# Compile model with gradient clipping
optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)  # Adam + gradient clipping
model.compile(optimizer=optimizer,  
              loss='sparse_categorical_crossentropy',  # Loss function : categorical crossentropy
              metrics=['accuracy'])

# Model summary
model.summary()

# Callbacks for learning rate reduction and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)  
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  

# Train model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=100, batch_size=128, verbose=1,  # Batch size at 128
                    callbacks=[reduce_lr, early_stop])

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# Save model
model.save('cldnn_model.keras')
