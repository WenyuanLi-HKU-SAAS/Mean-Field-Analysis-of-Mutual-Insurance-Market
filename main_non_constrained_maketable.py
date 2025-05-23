import numpy as np
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import pickle
import time
import matplotlib.pyplot as plt
class Parameters:
    # financial parameters
    a: float = 0.0
    b: float = 1.0
    r: float = 0.03
    l = tf.cast(np.array([[0.02, 0.02]]), dtype=tf.float32)
    P = tf.cast(np.array([[1.0, 1.0]]), dtype=tf.float32)
    mat_P = tf.linalg.diag(tf.squeeze(P))
    Q = tf.cast(np.array([[1.0, 1.0]]), dtype=tf.float32)
    mat_Q = tf.linalg.diag(tf.squeeze(Q))
    R = tf.cast(np.array([[0.1, 0.1]]), dtype=tf.float32)
    mat_R = tf.linalg.diag(tf.squeeze(R))
    S = tf.cast(np.array([[0.6, 0.6]]), dtype=tf.float32)
    mat_S = tf.linalg.diag(tf.squeeze(S))
    k = tf.cast(np.array([[0.5, 0.5]]), dtype=tf.float32)
    K = tf.linalg.diag(tf.squeeze(k))
    sig = tf.cast(np.array([[0.1, 0.3]]), dtype=tf.float32)
    Sig = tf.linalg.diag(tf.squeeze(sig))
    gamma = tf.cast(np.array([[1.0, 1.0]]), dtype=tf.float32)
    xi = tf.cast(np.array([[2.0, 2.0]]), dtype=tf.float32)
    omega = tf.cast(np.array([[0.5, 0.5]]), dtype=tf.float32)
    pi = tf.cast(np.array([[1.0, 1.0]]), dtype=tf.float32)
    lam = tf.constant(10.0)
    d: float = 0.05
    T: float = 1.0
    # path number
    N_1: int = 10000
    # time steps
    N_2: int = 101
    del_t: float = T / tf.cast(N_2-1, dtype=tf.float32)
    # Brownian motion
    np.random.seed(1)
    dW_t = tf.cast(np.random.standard_normal((N_1, N_2, 2))*np.sqrt(del_t),dtype=tf.float32)
    # Neural network parameters
    lr_init_values = 0.5*pow(10.0, -3.0)
    num_hiddens = np.array([30,30,1])
    num_iterations: int = 1000


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, para):
        super(FeedForwardSubNet, self).__init__()
        # 隐藏层个数
        self.dense_layers = [tf.keras.layers.Dense(para.num_hiddens[i],
                                                   use_bias=True,
                                                   activation=None,
                                                   kernel_regularizer='l1l2',
                                                   bias_regularizer='l1l2',
                                                   kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=pow(10.0,-3.0),seed=1),
                                                   bias_initializer=tf.random_normal_initializer(mean=0.7, stddev=pow(10.0, -2.0), seed=1),
                                                   )
                             for i in range(len(para.num_hiddens))]

    # 每一层做relu激活，最后一层直接输出
    def call(self, inputs):
        x = inputs
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        return x




class opt_loss(tf.keras.Model):
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.para.lr_init_values)
        # initialize rnn trainable variables
        # nn for v1_bar
        self.nn1 = FeedForwardSubNet(para)
        # nn for v2_bar
        self.nn2 = FeedForwardSubNet(para)
        # nn for eta1
        self.nn3 = FeedForwardSubNet(para)
        # nn for eta2
        self.nn4 = FeedForwardSubNet(para)
        # nn for p1
        self.nn5 = FeedForwardSubNet(para)
        # nn for p2
        self.nn6 = FeedForwardSubNet(para)

    def loss_fn(self):
        # initialization
        list_x1 = []
        list_x2 = []
        list_p1 = []
        list_p2 = []
        list_z1 = []
        list_z2 = []
        list_eta1 = []
        list_eta2 = []
        list_v1_bar = []
        list_v2_bar = []
        list_v1 = []
        list_v2 = []
        mat_one = tf.ones((2, 2))
        x1_now = tf.ones((self.para.N_1, 1)) * self.para.xi[0, 0]  # column is the path
        x2_now = tf.ones((self.para.N_1, 1)) * self.para.xi[0, 1]  # column is the path
        list_x1.append(x1_now)
        list_x2.append(x2_now)
        p1_now = self.nn5(x1_now)
        p2_now = self.nn6(x2_now)
        list_p1.append(p1_now)
        list_p2.append(p2_now)
        z1_now = tf.ones((1, 1)) * self.para.xi[0, 0]
        z2_now = tf.ones((1, 1)) * self.para.xi[0, 1]
        list_z1.append(z1_now)
        list_z2.append(z2_now)
        penalty = 0.00

        # iterate the time step
        for j in range(self.para.N_2 - 1):
            tnow = j * self.para.del_t

            # strategies
            # inputs = tf.cast(tnow * tf.ones([self.para.N_1, 1]), dtype=tf.float32)
            inputs = tf.cast(tnow * tf.ones([1, 1]), dtype=tf.float32)
            v1_bar = self.nn1(inputs)
            v2_bar = self.nn2(inputs)

            inputs = tf.concat([tf.cast(tnow * tf.ones([self.para.N_1, 1]), dtype=tf.float32), x1_now,
                                tf.ones((self.para.N_1, 1)) * z1_now, p1_now], axis=1)
            eta1 = self.nn3(inputs)
            inputs = tf.concat([tf.cast(tnow * tf.ones([self.para.N_1, 1]), dtype=tf.float32), x2_now,
                                tf.ones((self.para.N_1, 1)) * z2_now, p2_now], axis=1)
            eta2 = self.nn4(inputs)
            # vec_eta = tf.concat([eta1,eta2],axis=1)

            v1 = 1.0 / self.para.P[0, 0] * (self.para.k[0, 0] * p1_now + self.para.sig[0, 0] * eta1) + self.para.R[
                0, 0] * v1_bar
            # v1 = tf.clip_by_value(v1, clip_value_min=self.para.a, clip_value_max=self.para.b)
            v2 = 1.0 / self.para.P[0, 1] * (self.para.k[0, 1] * p2_now + self.para.sig[0, 1] * eta2) + self.para.R[
                0, 1] * v2_bar
            # v2 = tf.clip_by_value(v2, clip_value_min=self.para.a, clip_value_max=self.para.b)

            #weight1 = (self.para.N_2 - j) / (self.para.N_2)
            #weight2 = (self.para.N_2 - j) / (self.para.N_2)
            weight1 = 1.0
            weight2 = 1.0
            x1_now = x1_now + (self.para.r * x1_now + self.para.l[0, 0] - self.para.k[0, 0] * v1 +
                               (self.para.omega[0, 0] * (self.para.k[0, 0] - self.para.d) * v1_bar \
                                + self.para.omega[0, 1] * (self.para.k[0, 1] - self.para.d) * v2_bar) * self.para.pi[
                                   0, 0]) * self.para.del_t \
                     + self.para.sig[0, 0] * (1.0 - v1) * tf.reshape(self.para.dW_t[:, j, 0], (self.para.N_1, 1))

            p1_now = p1_now - (self.para.r * p1_now + (x1_now - z1_now * self.para.S[0, 0]) * self.para.Q[
                0, 0]) * self.para.del_t \
                     + eta1 * tf.reshape(self.para.dW_t[:, j, 0], (self.para.N_1, 1))

            z1_now = z1_now + (self.para.r * z1_now + self.para.l[0, 0] - v1_bar * self.para.k[0, 0] + (
                        self.para.omega[0, 0] * (self.para.k[0, 0] - self.para.d) * v1_bar \
                        + self.para.omega[0, 1] * (self.para.k[0, 1] - self.para.d) * v2_bar) * self.para.pi[
                                   0, 0]) * self.para.del_t

            penalty = penalty + tf.reduce_sum(weight1 * pow((tf.reduce_mean(v1) - v1_bar), 2.0))
            # print('j = ', j, 'penalty1 = ', weight1*pow((tf.reduce_mean(v1)-v1_bar),2.0))
            # print('j = ', j, 'variance1 = ', tf.math.reduce_variance(v1))

            x2_now = x2_now + (self.para.r * x2_now + self.para.l[0, 1] - self.para.k[0, 1] * v2 +
                               (self.para.omega[0, 0] * (self.para.k[0, 0] - self.para.d) * v1_bar \
                                + self.para.omega[0, 1] * (self.para.k[0, 1] - self.para.d) * v2_bar) * self.para.pi[
                                   0, 1]) * self.para.del_t \
                     + self.para.sig[0, 1] * (1.0 - v2) * tf.reshape(self.para.dW_t[:, j, 1], (self.para.N_1, 1))

            p2_now = p2_now - (self.para.r * p2_now + (x2_now - z2_now * self.para.S[0, 1]) * self.para.Q[
                0, 1]) * self.para.del_t \
                     + eta2 * tf.reshape(self.para.dW_t[:, j, 1], (self.para.N_1, 1))

            z2_now = z2_now + (self.para.r * z2_now + self.para.l[0, 1] - v2_bar * self.para.k[0, 1] + (
                        self.para.omega[0, 0] * (self.para.k[0, 0] - self.para.d) * v1_bar \
                        + self.para.omega[0, 1] * (self.para.k[0, 1] - self.para.d) * v2_bar) * self.para.pi[
                                   0, 1]) * self.para.del_t

            penalty = penalty + tf.reduce_sum(weight2 * pow((tf.reduce_mean(v2) - v2_bar), 2.0))
            # print('j = ', j, 'penalty2 = ', weight2*pow((tf.reduce_mean(v2) - v2_bar), 2.0))

            list_x1.append(x1_now)
            list_p1.append(p1_now)
            list_z1.append(z1_now)
            list_eta1.append(eta1)
            list_v1_bar.append(v1_bar)
            list_v1.append(v1)
            list_x2.append(x2_now)
            list_p2.append(p2_now)
            list_z2.append(z2_now)
            list_eta2.append(eta2)
            list_v2_bar.append(v2_bar)
            list_v2.append(v2)

        penalty = penalty / tf.cast(self.para.N_2, dtype=tf.float32)
        loss = tf.reduce_mean(
            pow(p1_now + self.para.gamma[0, 0] - (x1_now - z1_now * self.para.S[0, 0]) * self.para.Q[0, 0], 2.0) \
            + pow(p2_now + self.para.gamma[0, 1] - (x2_now - z2_now * self.para.S[0, 1]) * self.para.Q[0, 1],
                  2.0)) + self.para.lam * penalty
        return loss, penalty, list_x1, list_z1, list_p1, list_eta1, list_v1_bar, list_v1, list_x2, list_z2, list_p2, list_eta2, list_v2_bar, list_v2


class solver():
    def __init__(self, para):
        self.model = opt_loss(para)
        self.para = para
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.para.lr_init_values,
            decay_steps=self.para.num_iterations,
            decay_rate=0.96,
            staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    def train(self):
        start_time = time.time()
        iterations = []
        losses = []
        for step in range(self.para.num_iterations):
            # compute gradient
            with tf.GradientTape(persistent=True) as tape:
                loss, penalty, list_x1, list_z1, list_p1, list_eta1, list_v1_bar, list_v1, list_x2, list_z2, list_p2, list_eta2, list_v2_bar, list_v2 = self.model.loss_fn()
            grad = tape.gradient(loss, self.model.trainable_variables)

            # save intermediate result for loss figures
            elapsed_time = time.time() - start_time
            #print("step = ", step, "loss = ", loss.numpy(), "penalty = ", penalty.numpy(), "elapsed time = ", elapsed_time)

            # Update gradient
            self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
            iterations.append(step)
            losses.append(loss)

        loss, penalty, list_x1, list_z1, list_p1, list_eta1, list_v1_bar, list_v1, list_x2, list_z2, list_p2, list_eta2, list_v2_bar, list_v2 = self.model.loss_fn()
        # save the final result
        elapsed_time = time.time() - start_time
        #print("step = ", step, "loss = ", loss.numpy(), "penalty = ", penalty.numpy(), "elapsed time = ", elapsed_time)
        # compute ODE error
        with open('case1a_non_constrained_ODE.pickle', 'rb') as dualfile2:
            ODE_list_Gamma, ODE_list_Xi, ODE_list_zeta, ODE_list_z, ODE_list_pbar, ODE_list_vbar = pickle.load(
                dualfile2)

        ODE_list_vbar_1 = [row[0, 0] for row in ODE_list_vbar]
        ODE_list_vbar_2 = [row[1, 0] for row in ODE_list_vbar]
        ODE_list_z_1 = [row[0, 0] for row in ODE_list_z[:-1]]
        ODE_list_z_2 = [row[1, 0] for row in ODE_list_z[:-1]]

        total_error = tf.constant(0.0)
        for a_i, b_i in zip(list_v1_bar, ODE_list_vbar_1):
            total_error = total_error + tf.math.abs(a_i - b_i)
        for a_i, b_i in zip(list_v2_bar, ODE_list_vbar_2):
            total_error = total_error + tf.math.abs(a_i - b_i)
        for a_i, b_i in zip(list_z1, ODE_list_z_1):
            total_error = total_error + tf.math.abs(a_i - b_i)
        for a_i, b_i in zip(list_z2, ODE_list_z_2):
            total_error = total_error + tf.math.abs(a_i - b_i)
        total_error = total_error / tf.cast(self.para.N_2, dtype=tf.float32)

        return loss, penalty, total_error, elapsed_time




if __name__ == '__main__':
    para = Parameters()
    vec_lam = tf.cast(np.array([0.1,1.0,10.0,100.0,1000.0]),dtype=tf.float32)
    #vec_lam = tf.cast(np.array([0.1]), dtype=tf.float32)
    ans_list = np.zeros((4,5))
    for j in range(5):
        para.lam = vec_lam[j]
        MVBSDE_solver = solver(para)
        [loss, penalty, total_error, elapsed_time] = MVBSDE_solver.train()
        ans_list[0,j] = loss
        ans_list[1,j] = penalty
        ans_list[2,j] = total_error
        ans_list[3,j] = elapsed_time
        print("lambda = ", para.lam, "total error = ", total_error, "time_elapsed = ", elapsed_time)

    print("ans_list = ", ans_list)