import numpy as np
import timeit




def GD(f, grad_f, x_init, tau, iterMax, prec):

    epsilon = prec*np.linalg.norm(grad_f(x_init))

    x = np.copy(x_init)
    x_tab = np.copy(x_init)


    print("------------------------------------\n GD with constant step size\n------------------------------------\nSTART")
    t_s =  timeit.default_timer()

    for k in range(iterMax):

        g = grad_f(x)
        x = x - tau*g

        x_tab = np.vstack((x_tab,x))

        if np.linalg.norm(g) < epsilon:
            break

    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f} -- final gradient norm: {:f} \n\n".format(k,t_e-t_s,f(x),np.linalg.norm(grad_f(x))))
    
    return x,x_tab




def SGD(f, grad_f_subsampling, x_init, tau0, schedule, iterMax):
    # returns the sequence of iterates as well as the sequence of averaged iterates
    
    x = np.copy(x_init)
    x_tab = np.copy(x)
    tau = np.copy(tau0)

    x_avg = np.copy(x)
    x_avg_tab = np.copy(x_avg)
    x_sum = np.zeros(len(x_init))
    tau_sum = 0.0

    print("------------------------------------\n Stochastic gradient descent \n------------------------------------\nSTART")
    t_s =  timeit.default_timer()

    for k in range(iterMax):

        if schedule == "decreasing":
            tau = tau0 / (k+1)

        g = grad_f_subsampling(x)
        x = x - tau*g

        x_tab = np.vstack((x_tab,x))

        x_sum += tau*x
        tau_sum += tau
        x_avg = x_sum / tau_sum
        x_avg_tab = np.vstack((x_avg_tab,x_avg))


    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f}\n\n".format(k,t_e-t_s,f(x_avg)))
    
    return x,x_tab,x_avg, x_avg_tab


def SAGA(f, grad_f_i, x_init, tau0, iterMax, m):
    # returns the sequence of iterates as well as the sequence of averaged iterates
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
    t_s =  timeit.default_timer()

    for k in range(iterMax):
        i = np.random.randint(0, m)
        g_old = g_tab[i]
        g_tab[i] = grad_f_i(x, i)

        x = x - tau*(g_tab[i] - g_old + np.mean(g_tab, axis=0))
        x_tab = np.vstack((x_tab,x))

        x_sum += tau*x
        tau_sum += tau
        x_avg = x_sum / tau_sum
        x_avg_tab = np.vstack((x_avg_tab,x_avg))


    t_e =  timeit.default_timer()
    print("FINISHED -- {:d} iterations -- {:.6f}s -- final value: {:f}\n\n".format(k,t_e-t_s,f(x_avg)))
    
    return x,x_tab,x_avg, x_avg_tab



