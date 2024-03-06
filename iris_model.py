import torch
import numpy as np
import pandas as pd

# Set the training parameters
input_neuron = 4
output_neuron = 3
hidden_neuron = 20
max_epoch = 300
alpha = 0.0317

# ----------------------------

# Put the dataset to dataframe
df_raw = pd.read_csv("iris/iris.csv", header=None)
df = pd.get_dummies(df_raw, dtype=float)

# Rearrange the dataset
df_new = df.copy()
for i in range(50):
    df_new.loc[i*3] = df.loc[i]
    df_new.loc[i*3+1] = df.loc[i+50]
    df_new.loc[i*3+2] = df.loc[i+100]

df_in = df_new.iloc[:, 0:4]
df_out = df_new.iloc[:, 4:]

# Normalize the input
df_in = (df_in-df_in.min())/(df_in.max()-df_in.min())

# in_tensor = torch.cuda.DoubleTensor(df_in.values)
# out_tensor = torch.cuda.DoubleTensor(df_out.values)
in_tensor = torch.Tensor(df_in.values).cuda()
out_tensor = torch.Tensor(df_out.values).cuda()

# Split the training and testing set
in_train = in_tensor[0:100, :]
out_train = out_tensor[0:100, :]
in_test = in_tensor[100:, :]
out_test = out_tensor[100:, :]

train_length = len(in_train)
test_length = len(in_test)

# Inisialisasi bobot awal
uoj = torch.Tensor(np.random.uniform(-0.5, 0.5, (1, hidden_neuron))).cuda()
vok = torch.Tensor(np.random.uniform(-0.5, 0.5, (1, output_neuron))).cuda()

uij = torch.Tensor(np.random.uniform(-0.5, 0.5, (input_neuron, hidden_neuron))).cuda()
vjk = torch.Tensor(np.random.uniform(-0.5, 0.5, (hidden_neuron, output_neuron))).cuda()

# Inisialisasi Bobot Nguyen Widrow
beta_uij = 0.7 * (hidden_neuron)**(1.0/input_neuron)
beta_vjk = 0.7 * (output_neuron)**(1.0/hidden_neuron)

abs_uj = torch.sum(torch.mul(uij, uij), dim=0)
abs_vk = torch.sum(torch.mul(vjk, vjk), dim=0)

for j in range(len(abs_uj)):
    uij[:, j] = beta_uij * uij[:, j] / abs_uj[j]

for k in range(len(abs_vk)):
    vjk[:, k] = beta_vjk * vjk[:, k] / abs_vk[k]

# Proses Training
for epoch in range(max_epoch):
    # Inisialisasi ulang mse awal
    sum_mse = 0

    # Looping setiap vektor dataset
    for p in range(train_length):
        
        # START Proses Feed Forward
        # Input data ke hidden layer
        z_inj = torch.matmul(in_train[p, :].reshape(1, 4), uij) + uoj
        zj = torch.tanh(z_inj)

        # Input data ke output layer
        z_ink = torch.matmul(zj, vjk) + vok
        zk = torch.tanh(z_ink)

        # END

        # START Proses Back Propagation
        do_k = torch.mul(out_train[p, :].reshape(1, 3) - zk, 1.0 - torch.tanh(z_ink)**2)
        delta_vjk = alpha * torch.matmul(torch.t(zj), do_k)
        delta_vok = alpha * do_k

        do_inj = torch.Tensor(np.zeros((1, len(do_k)))).cuda()
        for j in range(len(do_k)):
            do_inj[j] = torch.sum(do_k * vjk[j, :])

        do_j = torch.mul(do_inj, 1.0 - torch.tanh(z_inj)**2)
        delta_uij = alpha * torch.matmul(in_train[p, :].reshape(1, 4).t(), do_j)
        delta_uoj = alpha * do_j

        # END

        # START Proses Pembaharuan Bobot
        uij = uij + delta_uij
        uoj = uoj + delta_uoj
        vjk = vjk + delta_vjk
        vok = vok + delta_vok

        # END

        # START Kalkulasi Error
        error = (out_train[p, :] - zk)**2
        sum_mse += error
        # END LOOP

    # START Print Evaluation Each Epoch
    print(f"Epoch {epoch} with mse of {sum_mse / train_length}")

    # END OF TRAINING

# TESTING
true_prediction = 0
false_prediction = 0
# Looping setiap vektor dataset
for p in range(test_length):
    
    # START Proses Feed Forward
    # Input data ke hidden layer
    z_inj = torch.matmul(in_test[p, :].reshape(1, 4), uij) + uoj
    zj = torch.tanh(z_inj)

    # Input data ke output layer
    z_ink = torch.matmul(zj, vjk) + vok
    zk = torch.tanh(z_ink)

    # Cek apakah output benar
    if torch.argmax(zk) == torch.argmax(out_test[p, :].reshape(1, 3)):
        true_prediction += 1
    else:
        false_prediction += 1

    # END

print(f"Training ended with true prediction = {true_prediction} and false prediction = {false_prediction}")