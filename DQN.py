# Basic DQN Module
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam

def load_dqn(state_dim, action_dim):
    # Input Layer
    inputs = Input(state_dim)

    # Hidden Layers
    x = Flatten()(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    
    # Output Layer
    predictions = Dense(action_dim, activation='linear')(x)

    # Build DQN model
    dqn_model = Model(inputs=inputs, outputs=predictions)
    learning_rate = 0.01
    optimizer = Adam(lr=learning_rate)
    dqn_model.compile(optimizer=optimizer, 
                  loss='mse', 
                  metrics=['accuracy'])

    return dqn_model