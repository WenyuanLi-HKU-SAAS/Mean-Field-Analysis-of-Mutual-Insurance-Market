import numpy as np
import tensorflow as tf
#import tensorflow_probability as tfp
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
import pickle
import time
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
class Parameters:
    # financial parameters
    a: float = 0.0
    b: float = 1.0
    r: float = 0.03
    l = tf.cast(np.array([[0.02,0.2]]),dtype=tf.float32)
    P = tf.cast(np.array([[1.0,1.0]]),dtype=tf.float32)
    mat_P = tf.linalg.diag(tf.squeeze(P))
    Q = tf.cast(np.array([[1.0,1.0]]),dtype=tf.float32)
    mat_Q = tf.linalg.diag(tf.squeeze(Q))
    R = tf.cast(np.array([[0.1,0.1]]),dtype=tf.float32)
    mat_R = tf.linalg.diag(tf.squeeze(R))
    S = tf.cast(np.array([[0.6,0.6]]),dtype=tf.float32)
    mat_S = tf.linalg.diag(tf.squeeze(S))
    k = tf.cast(np.array([[0.5,0.5]]),dtype=tf.float32)
    K = tf.linalg.diag(tf.squeeze(k))
    sig = tf.cast(np.array([[0.3,0.3]]),dtype=tf.float32)
    Sig = tf.linalg.diag(tf.squeeze(sig))
    gamma = tf.cast(np.array([[1.0,1.0]]),dtype=tf.float32)
    xi = tf.cast(np.array([[2.0,2.0]]),dtype=tf.float32)
    omega = tf.cast(np.array([[0.5,0.5]]),dtype=tf.float32)
    pi = tf.cast(np.array([[1.0,1.0]]),dtype=tf.float32)
    lam = tf.constant(1.0)
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
    num_hiddens = np.array([30,30,30,1])
    num_iterations: int = 1000
    max_range = 1.0


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, para):
        super(FeedForwardSubNet, self).__init__()
        #self.bn_layers = [
        #    tf.keras.layers.BatchNormalization(
        #        momentum=0.99,
        #        epsilon=1e-6,
        #        beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1,seed=1),
        #        gamma_initializer=tf.random_uniform_initializer(0.1, 0.5,seed=1)
        #    )
        #    for _ in range(len(para.num_hiddens) + 1)]
        # 隐藏层个数
        self.dense_layers = [tf.keras.layers.Dense(para.num_hiddens[i],
                                                   use_bias=True,
                                                   activation=None,
                                                   kernel_regularizer='l1l2',
                                                   bias_regularizer='l1l2',
                                                   kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=pow(10.0,-3.0),seed=1),
                                                   bias_initializer=tf.random_normal_initializer(mean=(para.b+para.a)/2.0,stddev=pow(10.0,-2.0),seed=1),
                                                   #bias_initializer='zeros'
                                                   )
                             for i in range(len(para.num_hiddens))]

    # 每一层做relu激活，最后一层直接输出
    def call(self, inputs):
        x = inputs
        #x = self.bn_layers[0](inputs,training=False)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            #x = self.bn_layers[i+1](x,training=False)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        #x = self.bn_layers[-1](x,training=False)
        return x


class opt_loss(tf.keras.Model):
    def __init__(self, para):
        super().__init__()
        self.para = para
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.para.lr_init_values)
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
        #initialization
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
        mat_one = tf.ones((2,2))
        x1_now = tf.ones((self.para.N_1,1))*self.para.xi[0,0]#column is the path
        x2_now = tf.ones((self.para.N_1,1))*self.para.xi[0,1]  # column is the path
        list_x1.append(x1_now)
        list_x2.append(x2_now)
        p1_0 = self.nn5(tf.ones([1, 1])*self.para.xi[0,0])
        p2_0 = self.nn6(tf.ones([1, 1])*self.para.xi[0,1])
        print('p1_0 = ', p1_0.numpy(), 'p2_0', p2_0.numpy())
        p1_now = tf.ones((self.para.N_1,1))*p1_0
        p2_now = tf.ones((self.para.N_1,1))*p2_0
        list_p1.append(p1_now)
        list_p2.append(p2_now)
        z1_now = tf.ones((1,1))*self.para.xi[0,0]
        z2_now = tf.ones((1,1))*self.para.xi[0,1]
        list_z1.append(z1_now)
        list_z2.append(z2_now)
        penalty = 0.00
        inputs = tf.cast(0.0 * tf.ones([1, 1]), dtype=tf.float32)
        v1_bar = tf.clip_by_value(self.nn1(inputs), clip_value_min=self.para.a, clip_value_max=self.para.b)
        v2_bar = tf.clip_by_value(self.nn2(inputs), clip_value_min=self.para.a, clip_value_max=self.para.b)
        print('v1_bar = ', v1_bar.numpy(), 'v2_bar', v2_bar.numpy())
        tnow = 0.0
        inputs = tf.concat(
            [tf.cast(tnow * tf.ones([self.para.N_1, 1]), dtype=tf.float32), tf.reshape(x1_now, (self.para.N_1, 1)),
             tf.ones((self.para.N_1, 1)) * z1_now, p1_now], axis=1)
        eta1 = self.nn3(inputs)
        inputs = tf.concat(
            [tf.cast(tnow * tf.ones([self.para.N_1, 1]), dtype=tf.float32), tf.reshape(x2_now, (self.para.N_1, 1)),
             tf.ones((self.para.N_1, 1)) * z2_now, p2_now], axis=1)
        eta2 = self.nn4(inputs)
        print('eta1 = ', eta1[0,0].numpy(), 'eta2', eta2[0,0].numpy())
        # iterate the time step
        for j in range(self.para.N_2 - 1):
            tnow = j * self.para.del_t

            # strategies
            #inputs = tf.cast(tnow * tf.ones([self.para.N_1, 1]), dtype=tf.float32)
            inputs = tf.cast(tnow * tf.ones([1, 1]), dtype=tf.float32)
            #v1_bar = tf.nn.sigmoid(self.nn1(inputs))*(self.para.b-self.para.a)+self.para.a
            v1_bar = tf.clip_by_value(self.nn1(inputs), clip_value_min=self.para.a, clip_value_max=self.para.b)
            v2_bar = tf.clip_by_value(self.nn2(inputs), clip_value_min=self.para.a, clip_value_max=self.para.b)

            inputs = tf.concat([tf.cast(tnow * tf.ones([self.para.N_1, 1]), dtype=tf.float32), tf.reshape(x1_now, (self.para.N_1, 1)), tf.ones((self.para.N_1, 1)) * z1_now, p1_now], axis=1)
            eta1 = self.nn3(inputs)
            inputs = tf.concat([tf.cast(tnow * tf.ones([self.para.N_1, 1]), dtype=tf.float32), tf.reshape(x2_now,(self.para.N_1,1)), tf.ones((self.para.N_1, 1)) * z2_now, p2_now],axis=1)
            eta2 = self.nn4(inputs)
            #vec_eta = tf.concat([eta1,eta2],axis=1)

            v1 = 1.0/self.para.P[0,0]*(self.para.k[0,0]*p1_now + self.para.sig[0,0]*eta1)+self.para.R[0,0]*v1_bar
            v1 = tf.clip_by_value(v1, clip_value_min=self.para.a, clip_value_max=self.para.b)
            v2 = 1.0/self.para.P[0,1]*(self.para.k[0,1]*p2_now + self.para.sig[0,1]*eta2)+self.para.R[0,1]*v2_bar
            v2 = tf.clip_by_value(v2, clip_value_min=self.para.a, clip_value_max=self.para.b)
            x1_now = x1_now + (self.para.r * x1_now + self.para.l[0,0] - self.para.k[0,0]*v1  +
                               (self.para.omega[0,0]*(self.para.k[0,0] - self.para.d)*v1_bar\
                                +self.para.omega[0,1]*(self.para.k[0,1] - self.para.d)*v2_bar)*self.para.pi[0,0]) * self.para.del_t \
                     + self.para.sig[0,0]*(1.0-v1)*tf.reshape(self.para.dW_t[:,j,0],(self.para.N_1,1))

            p1_now = p1_now - (self.para.r * p1_now + (x1_now - z1_now * self.para.S[0,0]) * self.para.Q[0,0]) * self.para.del_t \
                    + eta1*tf.reshape(self.para.dW_t[:,j,0],(self.para.N_1,1))

            z1_now = z1_now + (self.para.r * z1_now + self.para.l[0,0] - v1_bar * self.para.k[0,0] + (self.para.omega[0,0]*(self.para.k[0,0] - self.para.d)*v1_bar\
                                +self.para.omega[0,1]*(self.para.k[0,1] - self.para.d)*v2_bar)*self.para.pi[0,0]) * self.para.del_t

            penalty = penalty + tf.reduce_sum(pow(tf.reduce_mean(v1)-v1_bar,2.0))

            x2_now = x2_now + (self.para.r * x2_now + self.para.l[0,1] - self.para.k[0,1]*v2  +
                               (self.para.omega[0,0]*(self.para.k[0,0] - self.para.d)*v1_bar\
                                +self.para.omega[0,1]*(self.para.k[0,1] - self.para.d)*v2_bar)*self.para.pi[0,1]) * self.para.del_t \
                     + self.para.sig[0,1]*(1.0-v2)*tf.reshape(self.para.dW_t[:,j,1],(self.para.N_1,1))

            p2_now = p2_now - (self.para.r * p2_now + (x2_now - z2_now * self.para.S[0,1]) * self.para.Q[0,1]) * self.para.del_t \
                    + eta2*tf.reshape(self.para.dW_t[:,j,1],(self.para.N_1,1))

            z2_now = z2_now + (self.para.r * z2_now + self.para.l[0,1] - v2_bar * self.para.k[0,1] + (self.para.omega[0,0]*(self.para.k[0,0] - self.para.d)*v1_bar\
                                +self.para.omega[0,1]*(self.para.k[0,1] - self.para.d)*v2_bar)*self.para.pi[0,1]) * self.para.del_t

            penalty = penalty + tf.reduce_sum(pow(tf.reduce_mean(v2)-v2_bar,2.0))

            # gradient_F = -tf.matrix_diag(para.P)@(v-para.R@v_bar)


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
        penalty = penalty / tf.cast(self.para.N_2-1, dtype=tf.float32)
        loss = tf.reduce_mean(pow(p1_now + self.para.gamma[0,0]- (x1_now - z1_now * self.para.S[0,0] ) * self.para.Q[0,0],2.0)\
                            +pow(p2_now + self.para.gamma[0,1]- (x2_now - z2_now * self.para.S[0,1] ) * self.para.Q[0,1],2.0))+self.para.lam*penalty
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
            print("step = ", step, "loss = ", loss.numpy(),  "penalty = ", penalty.numpy(), "elapsed time = ", elapsed_time)

            # Update gradient
            self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
            iterations.append(step)
            losses.append(loss)

        loss, penalty, list_x1, list_z1, list_p1, list_eta1, list_v1_bar, list_v1, list_x2, list_z2, list_p2, list_eta2, list_v2_bar, list_v2 = self.model.loss_fn()
        # save the final result
        elapsed_time = time.time() - start_time
        print("step = ", step, "loss = ", loss.numpy(), "penalty = ", penalty.numpy(), "elapsed time = ", elapsed_time)
        iterations.append(step)
        losses.append(loss)
        print("Save result")
        with open('case1a_constrained.pickle', 'wb') as dualfile:
            pickle.dump([iterations, losses, elapsed_time, loss, penalty, list_x1, list_z1, list_p1, list_eta1, list_v1_bar, list_v1, list_x2, list_z2, list_p2, list_eta2, list_v2_bar, list_v2,  self.model.trainable_variables], dualfile)
        print("Result saved!")
        return loss

    def plot(self):

        with open('case1a_unconstrained.pickle', 'rb') as dualfile2:
            #iterations, losses, loss, penalty, list_x1, n_list_z1, list_p1, list_eta1, n_list_v1_bar, list_v1, list_x2, n_list_z2, list_p2, list_eta2, n_list_v2_bar, list_v2, train_var, elapsed_time  = pickle.load(dualfile2)
            loss, penalty, abs_error, rel_error, elapsed_time, n_list_v1_bar, n_list_v2_bar, n_list_z1, n_list_z2 = pickle.load(dualfile2)
        print('n_list_v1_bar0 = ', n_list_v1_bar[0].numpy(), 'n_list_v1_barT = ', n_list_v1_bar[-1].numpy())
        print('n_list_v2_bar0 = ', n_list_v2_bar[0].numpy(), 'n_list_v2_barT = ', n_list_v2_bar[-1].numpy())
        print('n_list_z1_bar0 = ', n_list_z1[0].numpy(), 'n_list_z1_barT = ', n_list_z1[-1].numpy())
        print('min_n_z1 = ', tf.reduce_min(n_list_z1).numpy(),'max_n_z1 = ', tf.reduce_max(n_list_z1).numpy())
        print('n_list_z2_bar0 = ', n_list_z2[0].numpy(), 'n_list_z2_barT = ', n_list_z2[-1].numpy())
        print('min_n_z2 = ', tf.reduce_min(n_list_z2).numpy(),'max_n_z2 = ', tf.reduce_max(n_list_z2).numpy())

        with open('case1a_constrained.pickle', 'rb') as dualfile1:
            c_iterations, c_losses, elapsed_time, loss, penalty, list_x1, c_list_z1, list_p1, list_eta1, c_list_v1_bar, list_v1, list_x2, c_list_z2, list_p2, list_eta2, c_list_v2_bar, list_v2, train_var = pickle.load(dualfile1)

        print('c_list_v1_bar0 = ', c_list_v1_bar[0].numpy(), 'c_list_v1_barT = ', c_list_v1_bar[-1].numpy())
        print('c_list_v2_bar0 = ', c_list_v2_bar[0].numpy(), 'c_list_v2_barT = ', c_list_v2_bar[-1].numpy())
        print('c_list_z1_0 = ', c_list_z1[0].numpy(), 'c_list_z1_T = ', c_list_z1[-1].numpy())
        print('min_c_z1 = ', tf.reduce_min(c_list_z1).numpy(),'max_c_z1 = ', tf.reduce_max(c_list_z1).numpy())
        print('c_list_z2_0 = ', c_list_z2[0].numpy(), 'c_list_z2_T = ', c_list_z2[-1].numpy())
        print('min_c_z2 = ', tf.reduce_min(c_list_z2).numpy(),'max_c_z2 = ', tf.reduce_max(c_list_z2).numpy())

        # 绘制迭代次数和损失函数曲线图
        #plt.figure(figsize=(8, 6))
        #plt.plot(c_iterations, c_losses, linewidth=2.5)
        #plt.xlabel('Iterations')
        #plt.ylabel('Loss')
        #plt.title('Training Loss Curve')
        #plt.show()

        vec_t = np.arange(0, self.para.N_2-1) * self.para.del_t
        plt.figure(figsize=(8, 6))
        n_list_v_bar_1 = [row[0] for row in n_list_v1_bar]
        plt.plot(vec_t, n_list_v_bar_1,color=[0, 0.4470, 0.7410],label=r'$\bar{v}^1_t(\gamma^1=0.5), w/o$',linewidth=2.5)
        n_list_v_bar_2 = [row[0] for row in n_list_v2_bar]
        plt.plot(vec_t, n_list_v_bar_2,color=[0.9290,0.6940,0.1250],label=r'$\bar{v}^2_t(\gamma^2=3.0), w/o$',linewidth=2.5)
        c_list_v_bar_1 = [row[0] for row in c_list_v1_bar]
        plt.plot(vec_t, c_list_v_bar_1,'-.',color=[0, 0.4470, 0.7410], label=r'$\bar{v}^1_t(\gamma^1=0.5), with$',linewidth=2.5)
        c_list_v_bar_2 = [row[0] for row in c_list_v2_bar]
        plt.plot(vec_t, c_list_v_bar_2, '-.', color=[0.9290,0.6940,0.1250],label=r'$\bar{v}^2_t(\gamma^2=3.0), with$',linewidth=2.5)
        #plt.legend(loc='upper left', bbox_to_anchor=(0.0, 0.7), prop = { "size": 12 })
        plt.legend(prop={"size": 12})
        plt.xlim(left=tf.reduce_min(vec_t))
        plt.xlim(right=tf.reduce_max(vec_t))
        #plt.ylim(bottom=-0.26443166)
        #plt.ylim(top=0.43274432)
        #Case 3
        #plt.ylim(bottom=-0.28056288)
        #plt.ylim(top=0.4323909)
        #Case 4
        #plt.ylim(bottom=0.00680643)
        #plt.ylim(top=0.5025271)
        #Case 5
        #plt.ylim(bottom=0.08968228)
        #plt.ylim(top=0.19359356)
        plt.xlabel(r'$t$')
        #plt.ylabel(r'$E[\theta_1]$')
        # plt.title('Expected theta')
        plt.show()

        vec_t = np.arange(0, self.para.N_2) * self.para.del_t
        plt.figure(figsize=(8, 6))
        n_list_z_1 = [row[0] for row in n_list_z1]
        plt.plot(vec_t, n_list_z_1,color=[0, 0.4470, 0.7410],label=r'$z^1_t(\gamma^1=0.5), w/o$',linewidth=2.5)
        n_list_z_2 = [row[0] for row in n_list_z2]
        plt.plot(vec_t, n_list_z_2,color=[0.9290,0.6940,0.1250],label=r'$z^2_t(\gamma^2=3.0), w/o$',linewidth=2.5)
        c_list_z_1 = [row[0] for row in c_list_z1]
        plt.plot(vec_t, c_list_z_1, '-.',color=[0, 0.4470, 0.7410], label=r'$z^1_t(\gamma^1=0.5), with$',linewidth=2.5)
        c_list_z_2 = [row[0] for row in c_list_z2]
        plt.plot(vec_t, c_list_z_2, '-.', color=[0.9290,0.6940,0.1250],label=r'$z^2_t(\gamma^2=3.0), with$',linewidth=2.5)
        plt.legend(prop = { "size": 12 },loc='upper left')
        plt.xlim(left=tf.reduce_min(vec_t))
        plt.xlim(right=tf.reduce_max(vec_t))
        #Case 2
        #plt.ylim(bottom=2.0)
        #plt.ylim(top=2.1472669)
        #Case 3
        #plt.ylim(bottom=1.998191)
        #plt.ylim(top=2.1305277)
        #Case 4
        #plt.ylim(bottom=1.9749559)
        #plt.ylim(top=2.2393277)
        #Case 5
        #plt.ylim(bottom=2.0)
        #plt.ylim(top=2.2682111)
        plt.xlabel(r'$t$')
        #plt.ylabel(r'$E[\theta_1]$')
        #plt.title('Expected theta')
        plt.show()


if __name__ == '__main__':
    para = Parameters()
    MVBSDE_solver = solver(para)
    #MVBSDE_solver.train()
    MVBSDE_solver.plot()