import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_alzheimers_model(input_dim):
    """
    Creates a Feed-Forward Neural Network for binary classification.
    """
    model = Sequential([
        # Input Layer
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        
        # Hidden Layers
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        
        # Output Layer (Binary Classification)
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
