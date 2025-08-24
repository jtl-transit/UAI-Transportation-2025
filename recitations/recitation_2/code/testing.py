import pandas as pd
from matplotlib import pyplot as plt
from util_nn_mlarch import dnn_alt_spec_estimation


#### import data
df = pd.read_csv('../data/mlogit_Train_wide.csv')
df_sp_train = pd.read_csv('../data/mlogit_Train_wide.csv')
df_sp_validation = pd.read_csv('../data/mlogit_Train_wide.csv')
df_sp_test = pd.read_csv('../data/mlogit_Train_wide.csv')

y_vars = ['choice']
z_vars = ['male', 'young_age', 'old_age', 'low_edu', 'high_edu',
         'low_inc', 'high_inc', 'full_job', 'age', 'inc', 'edu']
x0_vars = ['walk_walktime']
x1_vars = ['bus_cost', 'bus_walktime', 'bus_waittime', 'bus_ivt']
x2_vars = ['ridesharing_cost', 'ridesharing_waittime', 'ridesharing_ivt']
x3_vars = ['drive_cost', 'drive_walktime', 'drive_ivt']
x4_vars = ['av_cost', 'av_waittime', 'av_ivt']

X0_train = df_sp_train[x0_vars].values
X1_train = df_sp_train[x1_vars].values
X2_train = df_sp_train[x2_vars].values
X3_train = df_sp_train[x3_vars].values
X4_train = df_sp_train[x4_vars].values
Y_train = df_sp_train[y_vars].values.reshape(-1)
Z_train = df_sp_train[z_vars].values

X0_validation = df_sp_validation[x0_vars].values
X1_validation = df_sp_validation[x1_vars].values
X2_validation = df_sp_validation[x2_vars].values
X3_validation = df_sp_validation[x3_vars].values
X4_validation = df_sp_validation[x4_vars].values
Y_validation = df_sp_validation[y_vars].values.reshape(-1)
Z_validation = df_sp_validation[z_vars].values

X0_test = df_sp_test[x0_vars].values
X1_test = df_sp_test[x1_vars].values
X2_test = df_sp_test[x2_vars].values
X3_test = df_sp_test[x3_vars].values
X4_test = df_sp_test[x4_vars].values
Y_test = df_sp_test[y_vars].values.reshape(-1)
Z_test = df_sp_test[z_vars].values

# some hyperparameters
M_before = 5
M_after = 5
n_hidden_before = 40
n_hidden_after = 40
l1_const = 1e-10
l2_const = 1e-10
dropout_rate = 1e-50
batch_normalization = True 
learning_rate = 0.0001 
n_iterations = 5000 
n_mini_batch = 100

# one estimation here
train_accuracy,validation_accuracy,test_accuracy,prob_cost,prob_ivt = \
           dnn_alt_spec_estimation(X0_train,X1_train,X2_train,X3_train,X4_train,Y_train,Z_train,
                                   X0_validation,X1_validation,X2_validation,X3_validation,X4_validation,Y_validation,Z_validation,
                                   X0_test,X1_test,X2_test,X3_test,X4_test,Y_test,Z_test,
                                   M_before,M_after,n_hidden_before,n_hidden_after,l1_const,l2_const,
                                   dropout_rate,batch_normalization,learning_rate,n_iterations,n_mini_batch)

print("Training accuracy is ", train_accuracy)
print("Validation accuracy is ", validation_accuracy)
print("Testing accuracy is ", test_accuracy)
plt.plot(prob_cost[:, 3])
plt.plot(prob_ivt[:, 3])
