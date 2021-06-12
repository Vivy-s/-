import numpy as np
N = 3
Q = [1, 2, 3]
V = {'红':0, '白':1}
A = np.array([[0.5, 0.2, 0.3],
              [0.3, 0.5, 0.2],
              [0.2, 0.3, 0.5]])

B = np.array([[0.5, 0.5],
              [0.4, 0.6],
              [0.7, 0.3]])

pi = [0.2, 0.3, 0.5]
T = 8
O = [0, 1, 0, 0, 1, 0, 1, 1]
p = []

#前向算法
def forward(Q, V, A, B, pi, T, O, p):
    for t in range(T):
        # 计算初值
        if t == 0:
            for i in range(len(Q)):
                p.append(pi[i] * B[i, O[0]])
        # 初值计算完毕后，进行下一时刻的递推运算
        else:
            alpha_t_ = 0
            alpha_t_t = []
            for i in range(len(Q)):
                for j in range(len(Q)):
                    alpha_t_ += p[j] * A[j, i]
                alpha_t_t.append(alpha_t_ * B[i, O[t]])
                alpha_t_ = 0
            p = alpha_t_t
    return sum(p)
print('前向算法计算可得：')
print(forward(Q, V, A, B, pi, T, O, p))

#后向算法
def backward():
    beta = np.ones((T, N))
    for t in range(T-1):
        t = T - t - 2
        h = O[t + 1]
        h = int(h)
        for i in range(N):
            beta[t][i] = 0
            for j in range(N):
            # β(t,i)=和：a(i,j)*b(j,0(i))*β(t+1,j)
                beta[t][i] += A[i][j] * B[j][h] * beta[t+1][j]

    rate = 0
    for i in range(N):
        h = O[0]
        h = int(h)

        rate += pi[i] * B[i][h] * beta[0][i]
    return rate
print("后向算法计算可得Pb:\n", backward())
