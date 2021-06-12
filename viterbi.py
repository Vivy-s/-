import numpy as np
N = 3
Q = [1, 2, 3]
A = np.array([[0.5, 0.2, 0.3],
              [0.3, 0.5, 0.2],
              [0.2, 0.3, 0.5]])

B = np.array([[0.5, 0.5],
              [0.4, 0.6],
              [0.7, 0.3]])

pi = [0.2, 0.3, 0.5]
T = 8
p = []

#0代表红色，1代表白色
listA=[0,1,0,0,1,0,1,1]
o=np.array(listA)


def vitebi_decode(A,B,L,O):
    # 取出HMM模型的三个参数
    #A, B, L = hmm_paramter
    # 状态的总个数
    N = A.shape[0]
    # 获取观测序列的长度
    T = len(O)

    # 定义[N*T]的矩阵P，用来表示t时刻状态为i的最大概率
    P = np.zeros((N, T))
    # 定义[N*T]的矩阵D，用来表示t时刻状态为i取得最大概率时t-1时刻的状态
    D = np.zeros((N, T), dtype=int)
    # 初始化P,D在t=0时刻所对应的概率值和初状态
    for i in range(N):
        P[i][0] = L[i] * (B[i, O[0]])
        D[i][0] = -1

    # 接下来通过一个三层循环对P,D里面的元素进行赋值操作
    for i in range(1, T):
        obs_value = O[i]
        for j in range(N):
            pro_list = []
            for k in range(N):
                pro_list.append(P[k, i - 1] * A[k, j] * B[j, obs_value])
            max_value = max(pro_list)
            max_idx = pro_list.index(max_value)
            P[j, i] = max_value
            D[j, i] = max_idx

    # 获取P最后一个时刻的最大值，即为最大概率
    final_max_value = np.max(P, axis=0)[-1]
    final_max_idx = np.argmax(P, axis=0)[-1]

    # 通过finax_max_idx和D倒推出最大概率值所对应的状态序列
    state_list = []
    state_list.append(final_max_idx)
    for i in range(T - 1, 0, -1):
        final_max_idx = D[final_max_idx][i]
        state_list.append(final_max_idx)
    final_state_list = []

    # 由于状态是从1开始的，所以要对每一个状态索引加1
    for i in range(len(state_list) - 1, -1, -1):
        final_state_list.append(state_list[i] + 1)

    return final_max_value, final_state_list

max_rate, final_path=vitebi_decode(A,B,pi,o)

print('最优路径概率：')
print(max_rate)

print('\n最优路径：')
print(final_path)
