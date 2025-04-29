import numpy as np, pandas as pd, matplotlib.pyplot as plt, tensorflow.compat.v1 as tf
from sklearn.preprocessing import MinMaxScaler
tf.disable_v2_behavior()

# --- Load & visualize data ---
df = pd.read_csv("python_data.csv")
df = df[df.symbol == 'EQIX'].drop(['symbol', 'volume'], axis=1)

# --- Normalize ---
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df)

# --- Create sequences ---
def create_sequences(data, seq_len):
    sequences = [data[i:i+seq_len] for i in range(len(data) - seq_len)]
    sequences = np.array(sequences)
    return sequences[:,:-1], sequences[:,-1]

seq_len = 20
X, y = create_sequences(df.values, seq_len)
train_size = int(0.8 * len(X))
valid_size = int(0.1 * len(X))

x_train, y_train = X[:train_size], y[:train_size]
x_valid, y_valid = X[train_size:train_size+valid_size], y[train_size:train_size+valid_size]
x_test, y_test = X[train_size+valid_size:], y[train_size+valid_size:]

# --- RNN Parameters ---
n_steps, n_inputs = seq_len - 1, df.shape[1]
n_neurons, n_outputs, n_layers = 200, df.shape[1], 2
lr, batch_size, epochs = 0.001, 50, 100

# --- Build RNN Model ---
tf.reset_default_graph()
X_ph = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y_ph = tf.placeholder(tf.float32, [None, n_outputs])

cells = [tf.nn.rnn_cell.BasicRNNCell(n_neurons, activation=tf.nn.elu) for _ in range(n_layers)]
multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
rnn_outputs, _ = tf.nn.dynamic_rnn(multi_cell, X_ph, dtype=tf.float32)

outputs = tf.layers.dense(tf.reshape(rnn_outputs, [-1, n_neurons]), n_outputs)
outputs = tf.reshape(outputs, [-1, n_steps, n_outputs])[:, -1, :]

loss = tf.reduce_mean(tf.square(outputs - y_ph))
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

# --- Training ---
def get_batch(data_x, data_y):
    idx = np.random.randint(0, len(data_x), batch_size)
    return data_x[idx], data_y[idx]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    steps = int(len(x_train) * epochs / batch_size)
    for step in range(steps):
        x_batch, y_batch = get_batch(x_train, y_train)
        sess.run(train_op, feed_dict={X_ph: x_batch, y_ph: y_batch})
        if step % (steps // 10) == 0:
            l_train = sess.run(loss, {X_ph: x_train, y_ph: y_train})
            l_valid = sess.run(loss, {X_ph: x_valid, y_ph: y_valid})
            print(f"Step {step}: Train Loss={l_train:.6f}, Valid Loss={l_valid:.6f}")
    
    preds = lambda d: sess.run(outputs, {X_ph: d})
    y_train_pred, y_valid_pred, y_test_pred = map(preds, [x_train, x_valid, x_test])

# --- Plot Predictions ---
def plot_predictions(y_train, y_valid, y_test, y_train_pred, y_valid_pred, y_test_pred, feature=0):
    plt.figure(figsize=(14,5))
    all_true = np.concatenate([y_train, y_valid, y_test])[:,feature]
    all_pred = np.concatenate([y_train_pred, y_valid_pred, y_test_pred])[:,feature]
    plt.plot(all_true, label="True")
    plt.plot(all_pred, label="Predicted")
    plt.title("Stock Price Prediction")
    plt.legend(); plt.show()

plot_predictions(y_train, y_valid, y_test, y_train_pred, y_valid_pred, y_test_pred)

# --- Accuracy of Direction ---
def direction_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true[:,1]-y_true[:,0]) == np.sign(y_pred[:,1]-y_pred[:,0]))

acc_train = direction_accuracy(y_train, y_train_pred)
acc_valid = direction_accuracy(y_valid, y_valid_pred)
acc_test = direction_accuracy(y_test, y_test_pred)
print(f"Direction Accuracy (close-open): Train={acc_train:.2f}, Valid={acc_valid:.2f}, Test={acc_test:.2f}")