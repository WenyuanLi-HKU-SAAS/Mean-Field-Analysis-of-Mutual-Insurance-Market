import numpy as np
import tensorflow as tf
import pickle
import time
import matplotlib.pyplot as plt
class Parameters:
    # financial parameters
    r: float = 0.03
    l1_m_mu1 = tf.constant(0.02)
    l2_m_mu2 = tf.constant(0.02)
    e1 = tf.constant(0.01)
    e2 = tf.constant(0.01)
    e1_cost = tf.constant(0.9)
    e2_cost = tf.constant(0.9)
    omega = tf.cast(np.array([[0.5],[0.5]]),dtype=tf.float32)
    pi = tf.cast(np.array([[e1/(e1*omega[0,0]+e2*omega[1,0])],\
                            [e2/(e1*omega[0,0]+e2*omega[1,0])]]),dtype=tf.float32)
    print('pi = ', pi)
    l1 = l1_m_mu1 - e1 + pi[0,0]*(omega[0,0]*e1*e1_cost + omega[1,0]*e2*e2_cost)
    l2 = l2_m_mu2 - e2 + pi[1,0]*(omega[0,0]*e1*e1_cost + omega[1,0]*e2*e2_cost)
    l = tf.cast(np.array([[l1.numpy()],[l2.numpy()]]),dtype=tf.float32)
    print('l = ', l)
    mat_P = tf.cast(np.array([[1.0,0.0],[0.0,1.0]]),dtype=tf.float32)
    mat_Q = tf.cast(np.array([[1.0,0.0],[0.0,1.0]]),dtype=tf.float32)
    mat_R = tf.cast(np.array([[0.1,0.0],[0.0,0.1]]),dtype=tf.float32)
    mat_S = tf.cast(np.array([[0.6,0.0],[0.0,0.6]]),dtype=tf.float32)
    k = tf.cast(np.array([[0.5],[0.5]]),dtype=tf.float32)
    K = tf.linalg.diag(tf.squeeze(k))
    #omega = tf.cast(np.array([[0.5,0.6]]),dtype=tf.float32)
    sig = tf.cast(np.array([[0.1],[0.3]]),dtype=tf.float32)
    Sig = tf.linalg.diag(tf.squeeze(sig))
    gamma = tf.cast(np.array([[1.0],[1.0]]),dtype=tf.float32)
    x0 = tf.cast(np.array([[2.0],[2.0]]),dtype=tf.float32)
    d = 0.05
    Pi = tf.linalg.diag(tf.squeeze(pi)) @ np.array([[omega[0, 0] * (k[0, 0] - d), omega[1, 0] * (k[1, 0] - d)], [omega[0, 0] * (k[0, 0] - d), omega[1, 0] * (k[1, 0] - d)]])
    T: float = 1.0
    # time steps
    N_2: int = 101
    del_t: float = T / tf.cast(N_2-1, dtype=tf.float32)


class opt_loss(tf.keras.Model):
    def __init__(self, para):
        super().__init__()
        self.para = para

    def loss_fn(self):
        #initialization
        vec_one = tf.ones((2,1))
        mat_one = tf.constant([[1.0,0.0],[0.0,1.0]])
        Sig_square = self.para.Sig@self.para.Sig
        list_Gamma = []
        list_Xi = []
        list_zeta = []
        list_A = []
        list_b = []
        print("upper_bar = ", (1.0-self.para.mat_S[0,0])/2.0)
        #check condition
        mat_M = self.para.Pi@tf.linalg.inv(self.para.Pi-self.para.K)
        mat_one_minus_M = tf.transpose(mat_M)@mat_M
        mat_A = (mat_one_minus_M + tf.transpose(mat_one_minus_M))/2.0
        eigen_mat_A = tf.sqrt(tf.linalg.eigvals(mat_A))
        print("spectral norm of M = ", eigen_mat_A.numpy())

        mat_M = self.para.Pi@tf.linalg.inv(self.para.Pi-self.para.K)
        mat_one_minus_M = tf.transpose(mat_one - mat_M)@(mat_one - mat_M)
        mat_A = (mat_one_minus_M + tf.transpose(mat_one_minus_M))/2.0
        eigen_mat_A = tf.linalg.eigvals(mat_A)
        print("eigen_assumption51c = ", eigen_mat_A.numpy())

        mat_B = (mat_one-tf.transpose(self.para.Pi*tf.linalg.inv(self.para.Pi-self.para.K)))\
                @self.para.mat_Q@(mat_one-self.para.mat_S)
        mat_B = tf.transpose(mat_B)@mat_B
        mat_A = (mat_B+ tf.transpose(mat_B))/2.0
        eigen_mat_A = tf.linalg.eigvals(mat_A)
        print("eigen_assumption51d = ", eigen_mat_A.numpy())

        Gamma_now = tf.constant(self.para.mat_Q)
        list_Gamma.append(Gamma_now)
        Xi_now = tf.constant(self.para.mat_Q@(mat_one-self.para.mat_S))
        list_Xi.append(Xi_now)
        zeta_now = tf.constant(-self.para.gamma)
        list_zeta.append(zeta_now)
        # backward simulation
        for j in range(self.para.N_2-1):
            A_now = self.para.K@tf.linalg.inv(Sig_square@Gamma_now+self.para.mat_P@(mat_one-self.para.mat_R))
            #print('b_t = ', Sig_square@tf.linalg.inv(Sig_square@Gamma_now+self.para.mat_P@(mat_one-self.para.mat_R))@Gamma_now)
            b_now = tf.reshape(tf.linalg.diag_part(Sig_square@tf.linalg.inv(Sig_square@Gamma_now+self.para.mat_P@(mat_one-self.para.mat_R))@Gamma_now),(2,1))
            list_A.append(A_now)
            list_b.append(b_now)
            Gamma_now = Gamma_now + ( -self.para.K@self.para.K@Gamma_now@Gamma_now@tf.linalg.inv(Sig_square@Gamma_now+self.para.mat_P) \
                                      + 2.0*self.para.r*Gamma_now+self.para.mat_Q )*self.para.del_t
            zeta_now = zeta_now + ((self.para.r*vec_one+Xi_now@(self.para.Pi-self.para.K)@A_now)@zeta_now\
                                   +Xi_now@(self.para.l+(self.para.Pi-self.para.K))@b_now)*self.para.del_t
            Xi_now = Xi_now + (Xi_now@(self.para.Pi-self.para.K)@A_now@Xi_now\
                               +2.0*self.para.r*Xi_now + self.para.mat_Q@(mat_one-self.para.mat_S))*self.para.del_t

            list_Gamma.append(Gamma_now)
            list_zeta.append(zeta_now)
            list_Xi.append(Xi_now)

        A_now = self.para.K @ tf.linalg.inv(Sig_square @ Gamma_now + self.para.mat_P @ (mat_one - self.para.mat_R))
        b_now = tf.reshape(tf.linalg.diag_part(Sig_square@tf.linalg.inv(Sig_square@Gamma_now+self.para.mat_P@(mat_one-self.para.mat_R))@Gamma_now),(2,1))
        list_A.append(A_now)
        list_b.append(b_now)
        list_Gamma = tf.reverse(list_Gamma,axis=[0])
        list_zeta = tf.reverse(list_zeta,axis=[0])
        list_Xi = tf.reverse(list_Xi,axis=[0])
        list_A = tf.reverse(list_A,axis=[0])
        list_b = tf.reverse(list_b,axis=[0])


        list_z = []
        list_pbar = []
        list_vbar = []
        z_now = self.para.x0
        list_z.append(z_now)
        for j in range(self.para.N_2):
            Xi_now = list_Xi[j]
            zeta_now = list_zeta[j]
            A_now = list_A[j]
            b_now = list_b[j]
            pbar_now = Xi_now @ z_now + zeta_now
            list_pbar.append(pbar_now)
            vbar_now = A_now @ pbar_now + b_now
            list_vbar.append(vbar_now)
            z_now = z_now + (self.para.r * z_now + self.para.l + (self.para.Pi - self.para.K) @ vbar_now) * self.para.del_t
            list_z.append(z_now)

        return list_Gamma, list_Xi, list_zeta, list_z, list_pbar, list_vbar
class solver():
    def __init__(self, para):
        self.model = opt_loss(para)
        self.para = para

    def train(self):
        start_time = time.time()
        list_Gamma, list_Xi, list_zeta, list_z, list_pbar, list_vbar = self.model.loss_fn()
        # save the final result
        elapsed_time = time.time() - start_time
        print("elapsed time = ", elapsed_time)
        print("Save result")
        with open('case1a_non_constrained_ODE.pickle', 'wb') as dualfile:
            pickle.dump([list_Gamma, list_Xi, list_zeta, list_z, list_pbar, list_vbar], dualfile)
        print("Result saved!")

    def plot(self):
        with open('case1a_non_constrained_ODE.pickle', 'rb') as dualfile2:
            list_Gamma, list_Xi, list_zeta, list_z, list_pbar, list_vbar = pickle.load(dualfile2)

        print('list_vbar_0 = ', list_vbar[0].numpy())
        print('list_vbar_T = ', list_vbar[-1].numpy())
        print('list_pbar_0 = ', list_pbar[0].numpy())
        vec_t = np.arange(0, self.para.N_2) * self.para.del_t
        plt.figure(figsize=(8, 6))
        list_v_bar_1 = [row[0,0] for row in list_vbar]
        plt.plot(vec_t, list_v_bar_1,'b',label=r'$\bar{v}^1_t(\sigma^1=0.1)$',linewidth=2.0)
        list_v_bar_2 = [row[1,0] for row in list_vbar]
        plt.plot(vec_t, list_v_bar_2,color=[0.9290,0.6940,0.1250],label=r'$\bar{v}^2_t(\sigma^2=0.3)$',linewidth=2.0)
        plt.legend(prop = { "size": 15 })
        plt.xlim(left=tf.reduce_min(vec_t))
        plt.xlim(right=tf.reduce_max(vec_t))
        plt.xlabel(r'$t$')
        #plt.ylabel(r'$E[\theta_1]$')
        # plt.title('Expected theta')
        plt.show()

        vec_t = np.arange(0, self.para.N_2) * self.para.del_t
        plt.figure(figsize=(8, 6))
        list_z_1 = [row[0,0] for row in list_z[:-1]]
        plt.plot(vec_t, list_z_1,'b',label=r'$z^1_t(\sigma^1=0.1)$',linewidth=2.0)
        list_z_2 = [row[1,0] for row in list_z[:-1]]
        plt.plot(vec_t, list_z_2,color=[0.9290,0.6940,0.1250],label=r'$z^2_t(\sigma^2=0.3)$',linewidth=2.0)
        plt.legend(prop = { "size": 15 })
        plt.xlim(left=tf.reduce_min(vec_t))
        plt.xlim(right=tf.reduce_max(vec_t))
        plt.xlabel(r'$t$')
        #plt.ylabel(r'$E[\theta_1]$')
        # plt.title('Expected theta')
        plt.show()





if __name__ == '__main__':
    para = Parameters()
    MVBSDE_solver = solver(para)
    MVBSDE_solver.train()
    MVBSDE_solver.plot()