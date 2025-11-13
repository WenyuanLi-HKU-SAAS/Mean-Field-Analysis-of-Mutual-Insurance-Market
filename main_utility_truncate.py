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
    l1_m_mu1 = tf.constant(0.02)
    l2_m_mu2 = tf.constant(0.02)
    e1 = tf.constant(0.01)
    e2 = tf.constant(0.01)
    e1_cost = tf.constant(0.9)
    e2_cost = tf.constant(0.9)
    omega = tf.cast(np.array([[0.5,0.5]]),dtype=tf.float32)
    pi = tf.cast(np.array([[e1/(e1*omega[0][0]+e2*omega[0][1]),\
                            e2/(e1*omega[0][0]+e2*omega[0][1])]]),dtype=tf.float32)
    print('pi = ', pi)
    l1 = l1_m_mu1 - e1 + pi[0][0]*(omega[0][0]*e1*e1_cost + omega[0][1]*e2*e2_cost)
    l2 = l2_m_mu2 - e2 + pi[0][1]*(omega[0][0]*e1*e1_cost + omega[0][1]*e2*e2_cost)
    l = tf.cast(np.array([[l1,l2]]),dtype=tf.float32)
    print('l = ', l)

    P = tf.cast(np.array([[1.0,1.0]]),dtype=tf.float32)
    mat_P = tf.linalg.diag(tf.squeeze(P))
    Q = tf.cast(np.array([[1.0,1.0]]),dtype=tf.float32)
    mat_Q = tf.linalg.diag(tf.squeeze(Q))
    R = tf.cast(np.array([[0.1,0.1]]),dtype=tf.float32)
    mat_R = tf.linalg.diag(tf.squeeze(R))



    gamma = tf.cast(np.array([[0.5, 3.0]]), dtype=tf.float32)
    U_a = tf.cast(np.array([[1.0,1.0]]),dtype=tf.float32)
    U_b = tf.cast(np.array([[5.0,5.0]]),dtype=tf.float32)
    B = tf.cast(np.array([[2.5,2.5]]),dtype=tf.float32)

    k = tf.cast(np.array([[0.08,0.08]]),dtype=tf.float32)
    K = tf.linalg.diag(tf.squeeze(k))

    sig = tf.cast(np.array([[0.3,0.3]]),dtype=tf.float32)
    Sig = tf.linalg.diag(tf.squeeze(sig))
    xi = tf.cast(np.array([[2.0,2.0]]),dtype=tf.float32)

    lam = tf.constant(1.0)
    d = 0.05
    Pi = tf.linalg.diag(tf.squeeze(pi)) @ np.array([[omega[0, 0] * (k[0, 0] - d), omega[0, 1] * (k[0, 1] - d)], [omega[0, 0] * (k[0, 0] - d), omega[0, 1] * (k[0, 1] - d)]])
    T: float = 1.0
    # path number
    N_1: int = 10000
    # time steps
    N_2: int = 101
    del_t: float = T / tf.cast(N_2-1, dtype=tf.float32)
    # Brownian motion
    np.random.seed(1)
    dW_t = tf.cast(np.random.standard_normal((N_1, N_2, 2))*np.sqrt(del_t),dtype=tf.float32)
    #sampler = qmc.Sobol(d=N_2, scramble=False)
    #sampler.fast_forward(n=2 * N_2)
    #sample = sampler.random(n=N_1)
    #dW1_t = tf.cast(norm.ppf(sample)*np.sqrt(del_t),dtype=tf.float32)
    #dW2_t = dW1_t
    # Neural network parameters
    lr_init_values = 0.5*pow(10.0, -3.0)
    num_hiddens = np.array([32,32,1])
    num_iterations: int = 1000


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, para):
        super(FeedForwardSubNet, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(para.num_hiddens[i],
                                                   use_bias=True,
                                                   activation=None,
                                                   kernel_regularizer='l1l2',
                                                   bias_regularizer='l1l2',
                                                   kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=pow(10.0,-3.0),seed=1),
                                                   bias_initializer=tf.random_normal_initializer(mean=0.5,stddev=pow(10.0,-2.0),seed=1),
                                                   )
                             for i in range(len(para.num_hiddens))]

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
            v1_bar = tf.clip_by_value(self.nn1(inputs), clip_value_min=self.para.a, clip_value_max=self.para.b)
            v2_bar = tf.clip_by_value(self.nn2(inputs), clip_value_min=self.para.a, clip_value_max=self.para.b)

            inputs = tf.concat([tf.cast(tnow * tf.ones([self.para.N_1, 1]), dtype=tf.float32), x1_now,
                                tf.ones((self.para.N_1, 1)) * z1_now, p1_now], axis=1)
            eta1 = self.nn3(inputs)
            inputs = tf.concat([tf.cast(tnow * tf.ones([self.para.N_1, 1]), dtype=tf.float32), x2_now,
                                tf.ones((self.para.N_1, 1)) * z2_now, p2_now], axis=1)
            eta2 = self.nn4(inputs)

            v1 = 1.0 / self.para.P[0, 0] * (self.para.k[0, 0] * p1_now + self.para.sig[0, 0] * eta1) + self.para.R[
                0, 0] * v1_bar
            v1 = tf.clip_by_value(v1, clip_value_min=self.para.a, clip_value_max=self.para.b)
            v2 = 1.0 / self.para.P[0, 1] * (self.para.k[0, 1] * p2_now + self.para.sig[0, 1] * eta2) + self.para.R[
                0, 1] * v2_bar
            v2 = tf.clip_by_value(v2, clip_value_min=self.para.a, clip_value_max=self.para.b)

            # weight1 = (self.para.N_2 - j) / (self.para.N_2)
            # weight2 = (self.para.N_2 - j) / (self.para.N_2)
            weight1 = 1.0
            weight2 = 1.0
            x1_flag = tf.cast((x1_now>=0.0),dtype=tf.float32)
            x2_flag = tf.cast((x2_now>=0.0),dtype=tf.float32)

            p1_now = p1_now + (- self.para.r * p1_now - (x1_now - self.para.B[0, 0]) * self.para.Q[
                0, 0] + self.para.U_a[0, 0] * tf.pow(
                x1_flag * self.para.U_a[0, 0] * x1_now / self.para.gamma[0, 0] + self.para.U_b[0, 0],
                -self.para.gamma[0, 0])) * self.para.del_t \
                     + eta1 * tf.reshape(self.para.dW_t[:, j, 0], (self.para.N_1, 1))

            x1_now = x1_now + (self.para.r * x1_now + self.para.l[0, 0] - self.para.k[0, 0] * v1 +
                               (self.para.omega[0, 0] * (self.para.k[0, 0] - self.para.d) * v1_bar \
                                + self.para.omega[0, 1] * (self.para.k[0, 1] - self.para.d) * v2_bar) * self.para.pi[
                                   0, 0]) * self.para.del_t \
                     + self.para.sig[0, 0] * (1.0 - v1) * tf.reshape(self.para.dW_t[:, j, 0], (self.para.N_1, 1))

            z1_now = z1_now + (self.para.r * z1_now + self.para.l[0, 0] - v1_bar * self.para.k[0, 0] + (
                    self.para.omega[0, 0] * (self.para.k[0, 0] - self.para.d) * v1_bar \
                    + self.para.omega[0, 1] * (self.para.k[0, 1] - self.para.d) * v2_bar) * self.para.pi[
                                   0, 0]) * self.para.del_t

            penalty = penalty + tf.reduce_sum(weight1 * pow((tf.reduce_mean(v1) - v1_bar), 2.0))

            p2_now = p2_now + (- self.para.r * p2_now - (x2_now - self.para.B[0, 1]) * self.para.Q[
                0, 1] + self.para.U_a[0, 1] * tf.pow(
                x2_flag * self.para.U_a[0, 1] * x2_now / self.para.gamma[0, 1] + self.para.U_b[0, 1],
                -self.para.gamma[0, 1])) * self.para.del_t \
                     + eta2 * tf.reshape(self.para.dW_t[:, j, 1], (self.para.N_1, 1))

            x2_now = x2_now + (self.para.r * x2_now + self.para.l[0, 1] - self.para.k[0, 1] * v2 +
                               (self.para.omega[0, 0] * (self.para.k[0, 0] - self.para.d) * v1_bar \
                                + self.para.omega[0, 1] * (self.para.k[0, 1] - self.para.d) * v2_bar) * self.para.pi[
                                   0, 1]) * self.para.del_t \
                     + self.para.sig[0, 1] * (1.0 - v2) * tf.reshape(self.para.dW_t[:, j, 1], (self.para.N_1, 1))

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
        x1_flag = tf.cast((x1_now >= 0.0), dtype=tf.float32)
        x2_flag = tf.cast((x2_now >= 0.0), dtype=tf.float32)
        loss = tf.reduce_mean(
            tf.pow(p1_now \
                   + self.para.U_a[0, 0] * tf.pow(
                x1_flag * self.para.U_a[0, 0] * x1_now / self.para.gamma[0, 0] + self.para.U_b[0, 0],
                -self.para.gamma[0, 0]) \
                   - (x1_now - self.para.B[0, 0]) * self.para.Q[0, 0], 2.0) \
            + tf.pow(p2_now \
                     + self.para.U_a[0, 1] * tf.pow(
                x2_flag * self.para.U_a[0, 1] * x2_now / self.para.gamma[0, 1] + self.para.U_b[0, 1],
                -self.para.gamma[0, 1]) \
                     - (x2_now - self.para.B[0, 1]) * self.para.Q[0, 1], 2.0)) + self.para.lam * penalty
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
        with open('case5_constrained.pickle', 'wb') as dualfile:
            pickle.dump([iterations, losses, elapsed_time, loss, penalty, list_x1, list_z1, list_p1, list_eta1, list_v1_bar, list_v1, list_x2, list_z2, list_p2, list_eta2, list_v2_bar, list_v2,  self.model.trainable_variables], dualfile)
        print("Result saved!")
        return loss

    def plot(self):
        with open('case5_constrained.pickle', 'rb') as dualfile2:
            iterations, losses, elapsed_time, loss, penalty, list_x1, list_z1, list_p1, list_eta1, list_v1_bar, list_v1, list_x2, list_z2, list_p2, list_eta2, list_v2_bar, list_v2, train_var = pickle.load(dualfile2)

        print("lam = ", self.para.lam.numpy())
        print("loss = ", loss)
        print("penalty = ", penalty)
        print("elapsed_time = ", elapsed_time)
        #print("p_0 = ", list_p1[0],list_p2[0])

        z1_T = tf.reduce_mean(list_x1, axis=1)[-1]
        obj1_T = self.para.gamma[0,0]*list_x1[-1] - 0.5*self.para.Q[0,0]*pow(list_x1[-1] - self.para.B[0,0],2.0)
        E_obj1_T = tf.reduce_mean(obj1_T)

        z2_T = tf.reduce_mean(list_x2, axis=1)[-1]
        obj2_T = self.para.gamma[0, 0] * list_x2[-1] - 0.5 * self.para.Q[0, 0] * pow(
            list_x2[-1] - self.para.B[0, 1], 2.0)
        E_obj2_T = tf.reduce_mean(obj2_T)

        print('list_v1_bar0 = ', list_v1_bar[0].numpy(), 'list_v1_barT = ', list_v1_bar[-1].numpy())
        print('list_v2_bar0 = ', list_v2_bar[0].numpy(), 'list_v2_barT = ', list_v2_bar[-1].numpy())

        print('E_obj1_T = ',E_obj1_T,'E_obj2_T = ',E_obj2_T)
        print('list_v1_bar0 = ',list_v1_bar[0].numpy(),'list_v2_bar0 = ',list_v2_bar[0].numpy())

        print('z1_T = ', z1_T[-1].numpy(), 'x1_5q = ',np.percentile(list_x1[-1],5.0,axis=0), \
              'x1_95q = ', np.percentile(list_x1[-1], 95.0, axis=0), \
              'x1 gap = ', np.percentile(list_x1[-1], 95.0, axis=0)-np.percentile(list_x1[-1],5.0,axis=0))
        print('z2_T = ', z2_T[-1].numpy(), 'x2_5q = ', np.percentile(list_x2[-1], 5.0, axis=0), \
              'x2_95q = ', np.percentile(list_x2[-1], 95.0, axis=0), 'x2 gap = ',
              np.percentile(list_x2[-1], 95.0, axis=0) - np.percentile(list_x2[-1], 5.0, axis=0))


        # 绘制迭代次数和损失函数曲线图
        plt.figure(figsize=(8, 6))
        plt.plot(iterations, losses, linewidth=2.5)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        #plt.title('Training Loss Curve')
        plt.show()

        vec_t = np.arange(0, self.para.N_2) * self.para.del_t
        plt.figure(figsize=(8, 6))
        plt.plot(vec_t, tf.reduce_mean(list_x1,axis=1), 'r', linewidth=2.5)
        plt.plot(vec_t, np.percentile(list_x1,5.0,axis=1), 'b', linewidth=2.5)
        plt.plot(vec_t, np.percentile(list_x1,95.0,axis=1), 'b', linewidth=2.5)
        plt.xlim(left=tf.reduce_min(vec_t))
        plt.xlim(right=tf.reduce_max(vec_t))
        plt.ylim(bottom=1.579860)
        plt.ylim(top=2.578994)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$x^1_t$')
        #plt.title('Training Loss Curve')
        plt.show()

        vec_t = np.arange(0, self.para.N_2) * self.para.del_t
        plt.figure(figsize=(8, 6))
        plt.plot(vec_t, tf.reduce_mean(list_x2,axis=1), 'r', linewidth=2.5)
        plt.plot(vec_t, np.percentile(list_x2,5.0,axis=1), 'b', linewidth=2.5)
        plt.plot(vec_t, np.percentile(list_x2,95.0,axis=1), 'b', linewidth=2.5)
        plt.xlim(left=tf.reduce_min(vec_t))
        plt.xlim(right=tf.reduce_max(vec_t))
        plt.ylim(bottom=1.579860)
        plt.ylim(top=2.578994)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$x^2_t$')
        #plt.title('Training Loss Curve')
        plt.show()

        vec_t = np.arange(0, self.para.N_2-1) * self.para.del_t
        plt.figure(figsize=(8, 6))
        list_v_bar_1 = [row[0] for row in list_v1_bar]
        plt.plot(vec_t, list_v_bar_1,'b',label=r'$\bar{v}^1_t(l^1=0.02)$',linewidth=2.5)
        list_v_bar_2 = [row[0] for row in list_v2_bar]
        plt.plot(vec_t, list_v_bar_2,color=[0.9290,0.6940,0.1250],label=r'$\bar{v}^2_t(l^2=0.2)$',linewidth=2.5)
        plt.legend(prop = { "size": 15 })
        #plt.xlim(left=tf.reduce_min(vec_t))
        #plt.xlim(right=tf.reduce_max(vec_t))
        #plt.ylim(bottom=-0.06862408)
        #plt.ylim(top=0.40320972)
        plt.xlabel(r'$t$')
        #plt.ylabel(r'$E[\theta_1]$')
        # plt.title('Expected theta')
        plt.show()

        vec_t = np.arange(0, self.para.N_2) * self.para.del_t
        plt.figure(figsize=(8, 6))
        list_z_1 = [row[0] for row in list_z1]
        plt.plot(vec_t, list_z_1,'b',label=r'$z^1_t(l^1=0.02)$',linewidth=2.5)
        list_z_2 = [row[0] for row in list_z2]
        plt.plot(vec_t, list_z_2,color=[0.9290,0.6940,0.1250],label=r'$z^2_t(l^2=0.2)$',linewidth=2.5)
        plt.legend(prop = { "size": 15 },loc='upper left')
        #plt.xlim(left=tf.reduce_min(vec_t))
        #plt.xlim(right=tf.reduce_max(vec_t))
        #plt.ylim(bottom=2.0)
        #plt.ylim(top=2.2424707)
        plt.xlabel(r'$t$')
        #plt.ylabel(r'$E[\theta_1]$')
        # plt.title('Expected theta')
        plt.show()


        print('min_z1z2 = ', tf.reduce_min(tf.concat([list_z_1,list_z_2],axis=0)).numpy())
        print('max_z1z2 = ', tf.reduce_max(tf.concat([list_z_1,list_z_2],axis=0)).numpy())




if __name__ == '__main__':
    para = Parameters()
    MVBSDE_solver = solver(para)
    MVBSDE_solver.train()
    MVBSDE_solver.plot()