import numpy as np

T = 52
a = 0.15
alpha = 1.0
u_threshold = 6.0
x_threshold = 1.0

def u_from_x(x):
    u = np.zeros(T)
    for t in range(T):
        u[t] = (1+a)*x[t] - x[t+1]
    return u

def cumulated_catch(u):
    return sum([np.exp(-alpha*t)*u[t] for t in range(T)])

### TO BE COMPLETED
def split_z(z):
    x = z[:T + 1]
    u = z[T + 1:]
    return x, u


def merge_z(x, u):
    return np.concatenate((x, u))


def _build_dynamics_matrix():
    # B z = 0 encodes x_{t+1} - (1+a)x_t + u_t = 0 for t=0,...,T-1
    B = np.zeros((T, 2 * T + 1))
    for t in range(T):
        B[t, t] = -(1 + a) # x_t
        B[t, t + 1] = 1.0 # x_{t+1}
        B[t, (T + 1) + t] = 1.0 # u_t
    return B


B_DYN = _build_dynamics_matrix()
BBT_DYN = B_DYN @ B_DYN.T
W_CATCH = np.array([np.exp(-alpha * t) for t in range(T)])


def proj_dynamics(z):
    # Orthogonal projection onto affine subspace {z : B z = 0}
    residual = B_DYN @ z
    correction = B_DYN.T @ np.linalg.solve(BBT_DYN, residual)
    return z - correction


def proj_x(z):
    x, u = split_z(z)
    x_n = np.maximum(x, x_threshold)
    return merge_z(x_n, u)


def proj_periodic(z):
    x, u = split_z(z)
    x_n = np.copy(x)
    avg = 0.5 * (x_n[0] + x_n[-1])
    x_n[0] = avg
    x_n[-1] = avg
    return merge_z(x_n, u)


def proj_u(z):
    x, u = split_z(z)
    u_n = np.maximum(u, 0.0)
    return merge_z(x, u_n)


def proj_cumulated_catch(z):
    x, u = split_z(z)
    deficit = u_threshold - np.inner(W_CATCH, u)
    if deficit > 0:
        u = u + (deficit / np.inner(W_CATCH, W_CATCH)) * W_CATCH
    return merge_z(x, u)


def proj(z):
    # One full alternating-projection sweep.
    z_n = proj_dynamics(z)
    z_n = proj_x(z_n)
    z_n = proj_periodic(z_n)
    z_n = proj_u(z_n)
    z_n = proj_cumulated_catch(z_n)
    return z_n