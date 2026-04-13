import numpy as np
import timeit



from scipy.optimize import line_search
from scipy.optimize import minimize

def GD(f, f_grad, x_init, tau, iterMax, prec):

    epsilon = prec*np.linalg.norm(f_grad(x_init))

    x = np.copy(x_init)
    x_tab = np.copy(x_init)


    print("------------------------------------\n GD with constant step size\n------------------------------------\nSTART")
    t_s =  timeit.default_timer()

    for k in range(iterMax):

        g = f_grad(x)
        x = x - tau*g

        x_tab = np.vstack((x_tab,x))

        if np.linalg.norm(g) < epsilon:
            break

    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f} -- final gradient norm: {:f} \n\n".format(k,t_e-t_s,f(x),np.linalg.norm(f_grad(x))))
    
    return x,x_tab

def GD_wolfe(f , f_grad , x_init , prec, iterMax):
    
    x = np.copy(x_init)
    epsilon = prec*np.linalg.norm(f_grad(x_init) )
    x_tab = np.copy(x)

    print("------------------------------------\n Gradient with Wolfe line search\n------------------------------------\nSTART")
    t_s =  timeit.default_timer()

    for k in range(iterMax):
        g = f_grad(x)

        res = line_search(f, f_grad, x, -g, gfk=None, old_fval=None, old_old_fval=None, args=(), c1=0.0001, c2=0.9, amax=50)
        tau = res[0]


        x = x - tau*g 

        x_tab = np.vstack((x_tab,x))

        if np.linalg.norm(g) < epsilon:
            break

    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f} -- final gradient norm: {:f} \n\n".format(k,t_e-t_s,f(x),np.linalg.norm(f_grad(x))))
    
    return x,x_tab

def newton(f , f_grad_hessian , x_init , prec , iterMax ):
    x = np.copy(x_init)
    g,H = f_grad_hessian(x_init)
    epsilon = prec*np.linalg.norm(g)

    x_tab = np.copy(x)
    print("------------------------------------\nNewton's algorithm\n------------------------------------\nSTART")
    t_s =  timeit.default_timer()
    for k in range(iterMax):

        g,H = f_grad_hessian(x)
        x = x - np.linalg.solve(H,g)  

        x_tab = np.vstack((x_tab,x))

        if np.linalg.norm(g) < epsilon:
            break

    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f} -- final gradient norm: {:f} \n\n".format(k,t_e-t_s,f(x),np.linalg.norm(g)))
    return x,x_tab


def bfgs(f , f_grad , x_init , prec , iterMax ):

    x = np.copy(x_init)
    g = f_grad(x_init)
    epsilon = prec*np.linalg.norm(g)

    I = np.eye(len(x))
    W = np.copy(I)

    x_tab = np.copy(x)
    print("------------------------------------\nBFGS algorithm\n------------------------------------\nSTART")
    t_s =  timeit.default_timer()

    for k in range(iterMax):

        d = W@g

        res = line_search(f, f_grad, x, -d, gfk=None, old_fval=None, old_old_fval=None, args=(), c1=0.0001, c2=0.9, amax=50)
        tau = res[0]

        x_new = x - tau*d
        g_new = f_grad(x_new)
        s = -tau*d
        y = g_new-g

        # TO DO: UPDATE THE MATRIX W
        W = (I-np.outer(s,y)/np.inner(y,s)) @ W @ (I-np.outer(y,s)/np.inner(y,s)) + np.outer(s,s)/np.inner(y,s)
        
        x = x_new
        g = g_new


        x_tab = np.vstack((x_tab,x))

        if np.linalg.norm(g) < epsilon:
            break

    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f} -- final gradient norm: {:f} \n\n".format(k,t_e-t_s,f(x),np.linalg.norm(g)))
    return x,x_tab

def newton_ls(f , f_grad_hessian , x_init , prec , iterMax ):
    """ Newton with Wolfe line search"""
    x = np.copy(x_init)
    g,H = f_grad_hessian(x_init)
    epsilon = prec*np.linalg.norm(g)

    x_tab = np.copy(x)
    print("------------------------------------\nNewton's algorithm\n------------------------------------\nSTART")
    t_s =  timeit.default_timer()
    for k in range(iterMax):

        g,H = f_grad_hessian(x)
        dir = -np.linalg.solve(H,g)

        res = line_search(f, lambda x : f_grad_hessian(x)[0], x, dir, gfk=None, old_fval=None, old_old_fval=None, args=(), c1=0.0001, c2=0.9, amax=50)
        tau = res[0]
        x = x + tau*dir   

        x_tab = np.vstack((x_tab,x))

        if np.linalg.norm(g) < epsilon:
            break

    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f} -- final gradient norm: {:f} \n\n".format(k,t_e-t_s,f(x),np.linalg.norm(g)))
    return x,x_tab

def GD_accelerated(f, grad_f, x_init, tau, iterMax, prec, c=0.5):

    epsilon = prec*np.linalg.norm(grad_f(x_init))

    x = np.copy(x_init)
    x_tab = np.copy(x_init)

    y = np.copy(x_init)
    lmbd = 0.0


    print("------------------------------------\n Accelerated GD with constant step size\n------------------------------------\nSTART")
    t_s =  timeit.default_timer()

    for k in range(iterMax):

        g = grad_f(y)
        x_new = y - tau * g
        lmbd_new = (1 + np.sqrt(1 + 4 * lmbd**2)) / 2
        y = x_new + (lmbd - 1) / lmbd_new * (x_new - x)
        x = x_new
        lmbd = lmbd_new

        x_tab = np.vstack((x_tab,x))

        if np.linalg.norm(g) < epsilon:
            break

    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f} -- final gradient norm: {:f} \n\n".format(k,t_e-t_s,f(x),np.linalg.norm(grad_f(x))))
    
    return x,x_tab



def CG_quadratic(A, b, f, grad_f, x_init, iterMax, prec):

    epsilon = prec*np.linalg.norm(grad_f(x_init))

    x = np.copy(x_init)
    x_tab = np.copy(x_init)

    r = -(A@x + b)
    d = r

    print("------------------------------------\n CG for quadratic objective \n------------------------------------\nSTART")
    t_s =  timeit.default_timer()

    for k in range(iterMax):

        tau_k = (r @ d) / (d @ (A @ d))
        x = x + tau_k * d
        r_new = -(A @ x + b)
        beta_k = (r_new @ r_new) / (r @ r)
        d = r_new + beta_k * d
        r = r_new

        x_tab = np.vstack((x_tab, x))

        if np.linalg.norm(grad_f(x)) < epsilon:
            break


    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f} -- final gradient norm: {:f} \n\n".format(k,t_e-t_s,f(x),np.linalg.norm(grad_f(x))))
    
    return x,x_tab



def CG_nonLinear(f, grad_f, x_init, iterMax, prec, tau0, rho, c):

    epsilon = prec*np.linalg.norm(grad_f(x_init))

    x = np.copy(x_init)
    x_tab = np.copy(x_init)

    r = -grad_f(x)
    d = r

    print("------------------------------------\n CG for quadratic objective \n------------------------------------\nSTART")
    t_s =  timeit.default_timer()

    for k in range(iterMax):

        # Armijo backtracking line search along direction d
        tau = tau0
        while f(x + tau * d) > f(x) + c * tau * (grad_f(x) @ d):
            tau = rho * tau

        x = x + tau * d
        r_new = -grad_f(x)
        beta_k = max((r_new @ (r_new - r)) / (r @ r), 0.0)
        d = r_new + beta_k * d
        r = r_new

        x_tab = np.vstack((x_tab, x))

        if np.linalg.norm(grad_f(x)) < epsilon:
            break


    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f} -- final gradient norm: {:f} \n\n".format(k,t_e-t_s,f(x),np.linalg.norm(grad_f(x))))
    
    return x,x_tab


def BFGS(f, grad_f, x_init):
    res = minimize(f, x_init, method='BFGS', jac=grad_f, options={'disp': True})
    return res.x


def Armijo(f, gradf, x, tau, c, rho):
    while f(x - tau * gradf(x)) > (f(x) + c * tau * np.linalg.norm(gradf(x))**2):
        tau = rho * tau
    return tau


def GD_ls(f, grad_f, x_init, tau0, iterMax, prec, rho, c):

    epsilon = prec * np.linalg.norm(grad_f(x_init))

    x = np.copy(x_init)
    x_tab = np.copy(x_init)

    print("------------------------------------\n GD with constant step size\n------------------------------------\nSTART")
    t_s = timeit.default_timer()

    for k in range(iterMax):

        g = grad_f(x)
        tau = Armijo(f, grad_f, x, tau0, c, rho)

        x = x - tau * g

        x_tab = np.vstack((x_tab, x))

        if np.linalg.norm(g) < epsilon:
            break

    t_e = timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f} -- final gradient norm: {:f} \n\n".format(k, t_e - t_s, f(x), np.linalg.norm(grad_f(x))))

    return x, x_tab


def GD_sec_order_exact(f, grad_f, hessian_f, x_init, iterMax, prec):

    epsilon = prec * np.linalg.norm(grad_f(x_init))

    x = np.copy(x_init)
    x_tab = np.copy(x_init)

    print("------------------------------------\n GD with constant step size\n------------------------------------\nSTART")
    t_s = timeit.default_timer()

    for k in range(iterMax):

        g = grad_f(x)
        A = hessian_f(x)
        tau = (g.T @ g) / (g.T @ A @ g)

        x = x - tau * g

        x_tab = np.vstack((x_tab, x))

        if np.linalg.norm(g) < epsilon:
            break

    t_e = timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f} -- final gradient norm: {:f} \n\n".format(k, t_e - t_s, f(x), np.linalg.norm(grad_f(x))))

    return x, x_tab


def SGD(f, grad_f_subsampling, x_init, tau0, schedule, iterMax):
    x = np.copy(x_init)
    x_tab = np.copy(x)
    tau = np.copy(tau0)

    x_avg = np.copy(x)
    x_avg_tab = np.copy(x_avg)
    x_sum = np.zeros(len(x_init))
    tau_sum = 0.0

    print("------------------------------------\n Stochastic gradient descent \n------------------------------------\nSTART")
    t_s = timeit.default_timer()

    for k in range(iterMax):

        if schedule == "decreasing":
            tau = tau0 / (k + 1)

        g = grad_f_subsampling(x)
        x = x - tau * g

        x_tab = np.vstack((x_tab, x))

        x_sum += tau * x
        tau_sum += tau
        x_avg = x_sum / tau_sum
        x_avg_tab = np.vstack((x_avg_tab, x_avg))

    t_e = timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f}\n\n".format(k, t_e - t_s, f(x_avg)))

    return x, x_tab, x_avg, x_avg_tab


def SAGA(f, grad_f_i, x_init, tau0, iterMax, m):
    n = x_init.size
    x = np.copy(x_init)
    x_tab = np.copy(x)
    tau = np.copy(tau0)

    x_avg = np.copy(x)
    x_avg_tab = np.copy(x_avg)
    x_sum = np.zeros(len(x_init))
    tau_sum = 0.0

    g_tab = np.zeros((m, n))
    for i in range(m):
        g_tab[i] = grad_f_i(x, i)

    print("------------------------------------\n SAGA \n------------------------------------\nSTART")
    t_s = timeit.default_timer()

    for k in range(iterMax):
        i = np.random.randint(0, m)
        g_old = g_tab[i]
        g_tab[i] = grad_f_i(x, i)

        x = x - tau * (g_tab[i] - g_old + np.mean(g_tab, axis=0))
        x_tab = np.vstack((x_tab, x))

        x_sum += tau * x
        tau_sum += tau
        x_avg = x_sum / tau_sum
        x_avg_tab = np.vstack((x_avg_tab, x_avg))

    t_e = timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f}\n\n".format(k, t_e - t_s, f(x_avg)))

    return x, x_tab, x_avg, x_avg_tab


def adagrad_norm(f, grad_f_subsampling, x_init, tau, b_sq, iterMax):

    x = np.copy(x_init)
    x_tab = np.copy(x)

    Gk = b_sq

    print("------------------------------------\n Adagrad-norm \n------------------------------------\nSTART")
    t_s = timeit.default_timer()

    for k in range(iterMax):

        gk = grad_f_subsampling(x)
        Gk += np.dot(gk, gk)
        x = x - (tau / np.sqrt(Gk)) * gk

        x_tab = np.vstack((x_tab, x))

    t_e = timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f}\n\n".format(k, t_e - t_s, f(x)))

    return x, x_tab


def adagrad_diag(f, grad_f_subsampling, x_init, tau, b_sq, iterMax):

    x = np.copy(x_init)
    x_tab = np.copy(x)
    n = x_init.size

    Gk = b_sq * np.identity(n)

    print("------------------------------------\n Adagrad \n------------------------------------\nSTART")
    t_s = timeit.default_timer()

    for k in range(iterMax):

        gk = grad_f_subsampling(x)
        Gk += np.diag(gk * gk)
        x = x - tau * np.linalg.inv(np.sqrt(Gk)) @ gk

        x_tab = np.vstack((x_tab, x))

    t_e = timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f}\n\n".format(k, t_e - t_s, f(x)))

    return x, x_tab


def adam(f, grad_f_subsampling, x_init, tau, beta1, beta2, delta, iterMax):

    x = np.copy(x_init)
    x_tab = np.copy(x)
    n = x_init.size

    mk = np.zeros(n)
    vk = np.zeros(n)

    print("------------------------------------\n Adam \n------------------------------------\nSTART")
    t_s = timeit.default_timer()

    for k in range(iterMax):

        gk = grad_f_subsampling(x)
        mk = beta1 * mk + (1 - beta1) * gk
        vk = beta2 * vk + (1 - beta2) * gk * gk
        mk_hat = mk / (1 - beta1**(k + 1))
        vk_hat = vk / (1 - beta2**(k + 1))
        x = x - tau * np.linalg.inv(np.sqrt(delta * np.identity(n) + np.diag(vk_hat))) @ mk_hat
        x_tab = np.vstack((x_tab, x))

    t_e = timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f}\n\n".format(k, t_e - t_s, f(x)))

    return x, x_tab


def proj_GD(f, grad_f, proj, x_init, tau, iterMax, prec):

    epsilon = prec * np.linalg.norm(grad_f(x_init))

    x = np.copy(x_init)
    x_tab = np.copy(x_init)

    print("------------------------------------\n GD with constant step size\n------------------------------------\nSTART")
    t_s = timeit.default_timer()

    for k in range(iterMax):

        x = proj(x - tau * grad_f(x))

        x_tab = np.vstack((x_tab, x))

        if np.linalg.norm(x_tab[-1] - x_tab[-2]) < epsilon:
            break

    t_e = timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f} -- final gradient norm: {:f} \n\n".format(k, t_e - t_s, f(x), np.linalg.norm(grad_f(x))))

    return x, x_tab


def POCS(proj, x_init, iterMax):
    x = np.copy(x_init)
    x_tab = np.copy(x_init)

    print("------------------------------------\n POCS \n------------------------------------\nSTART")
    t_s = timeit.default_timer()

    for k in range(iterMax):

        x = proj(x)

        x_tab = np.vstack((x_tab, x))

    t_e = timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s \n\n".format(k, t_e - t_s))

    return x, x_tab


def prox_grad(F, grad_f, prox_g, x_init, tau, iterMax, prec):

    epsilon = prec

    x = np.copy(x_init)
    x_tab = np.copy(x_init)

    print("------------------------------------\n prox gradient with constant step size\n------------------------------------\nSTART")
    t_s = timeit.default_timer()

    for k in range(iterMax):
        x = prox_g(x - tau * grad_f(x), tau)
        x_tab = np.vstack((x_tab, x))
        if np.linalg.norm(x_tab[k] - x_tab[k + 1]) < epsilon:
            break

    t_e = timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f} \n\n".format(k, t_e - t_s, F(x)))

    return x, x_tab

