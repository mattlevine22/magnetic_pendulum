import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import Ridge
from pdb import set_trace as bp

def my_solve_ivp(ic, f_rhs, t_eval, t_span, settings):
    u0 = np.copy(ic)
    if settings['method']=='Euler':
        dt = settings['dt']
        u_sol = np.zeros((len(t_eval), len(ic)))
        u_sol[0] = u0
        for i in range(len(t_eval)-1):
            t = t_eval[i]
            rhs = f_rhs(t, u0)
            u0 += dt * rhs
            u_sol[i] = u0
    else:
        sol = solve_ivp(fun=lambda t, y: f_rhs(y, t), t_span=t_span, y0=u0, t_eval=t_eval, **settings)
        u_sol = sol.y.T
    return np.squeeze(u_sol)

def my_regressor(X, y, alpha=1e-5, positive=False):

    alpha_effective = alpha * X.shape[0] / X.shape[1]
    print('alpha_effective = alpha * n_data / X.shape[1] = {} * {} / {} = {}'.format(alpha, X.shape[0], X.shape[1], alpha_effective))
    clf = Ridge(alpha=alpha_effective, fit_intercept=False, positive=positive)
    # note on solvers: solver{‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’, ‘lbfgs’}, default=’auto’
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

    clf.fit(X, y)
    y_pred = clf.predict(X)
    residuals = y_pred - y

    mse = np.mean((residuals)**2)
    r2 = clf.score(X,y)

    print('Training MSE:', mse)
    print('Training R^2:', r2)

    return clf, y_pred, residuals, mse, r2



class MagneticPendulum:
    """"""
    def __init__(_s, omega=0.5, alpha=0.2, h=0.2, do_normalization=True):
        _s.omega2 = omega**2
        _s.alpha = alpha
        _s.h = h
        _s.loc = np.array([[1/np.sqrt(3), 0], [-1/ (2*np.sqrt(3)), -0.5], [-1/ (2*np.sqrt(3)), 0.5]])
        _s.n_loc = _s.loc.shape[0]
        _s.do_normalization = do_normalization

    def get_ic(_s, box_size=2.0):
        return np.random.uniform(low=-box_size, high=box_size, size=(2*_s.loc.shape[1]))

    def make_grid(_s, box_size=1, n_per_axis=10):
        x = np.linspace(start=-box_size, stop=box_size, num=n_per_axis)
        u_all = np.array(np.meshgrid(x, x, x, x)).T.reshape(-1, 4)
        return u_all

    def Dsums(_s, x2, y2):
        xSum = 0
        ySum = 0
        for j in range(_s.n_loc):
            Dj3 = _s.D(_s.loc[j,0], _s.loc[j,1], x2, y2)**3
            xSum_j = (_s.loc[j,0] - x2) / Dj3
            ySum_j = (_s.loc[j,1] - y2) / Dj3

            xSum += xSum_j
            ySum += ySum_j

        return xSum, ySum

    def D(_s, x_loc, y_loc, x, y):
        foo = (x_loc - x)**2 + (y_loc - y)**2 + _s.h
        foo = np.sqrt(foo)
        return foo

    def get_inits(_s):

        return state_inits

    def rhs(_s, u, t=0):
        ''' Full system RHS '''
        u = u.T
        udot = np.zeros_like(u)
        x1 = u[0]
        x2 = u[1]
        y1 = u[2]
        y2 = u[3]

        Dsums_x, Dsums_y = _s.Dsums(x2, y2)

        udot[0] = -_s.omega2*x2 - _s.alpha*x1 + Dsums_x
        udot[1] = x1
        udot[2] = -_s.omega2*y2 - _s.alpha*y1 + Dsums_y
        udot[3] = y1

        return udot.T

    def rhs_approx(_s, u, t=0):
        if u.ndim==1:
            u = u.reshape(1,-1)
        F = _s.compute_feature_maps(_s.scaleX(u))
        udot = _s.descaleY(_s.clf.predict(F))
        return udot

    def make_random_features(_s, n_input_dim, n_features):
        '''n_input_dim [scalar]: This is the dimension of the input space.
           n_features [scalar]: This is the number of random features to generate.
        '''
        # Generate random weights and biases for features
        _s.A = np.random.randn(n_features, n_input_dim)
        _s.b = np.random.uniform(low=0, high=2*np.pi, size=n_features)

    def compute_feature_maps(_s, u_input):
        return np.cos(_s.fac_A * (_s.A @ u_input.T).T + _s.b)

    def train_random_features(_s, n_features, u_input, u_output, alpha=1e-4, fac_A=1):
        n_input_dim = u_input.shape[1]
        _s.make_random_features(n_input_dim, n_features)
        _s.fac_A = fac_A

        F = _s.compute_feature_maps(_s.scaleX(u_input, save=True))
        print('F.shape:', F.shape)

        _s.clf, y_pred, residuals, _s.train_mse, _s.train_r2 = my_regressor(F, _s.scaleY(u_output, save=True), alpha=alpha)
        _s.coef_mean = np.mean(_s.clf.coef_)
        _s.coef_std = np.std(_s.clf.coef_)

        print('mean of coeffs:', _s.coef_mean)
        print('sd of coeffs:', _s.coef_std)

    ## define normalizer tools
    def scaleX(_s, x, save=False):
        if _s.do_normalization:
            if save:
                _s.x_mean = np.mean(x)
                _s.x_std = np.std(x)
            return (x-_s.x_mean) / _s.x_std
        else:
            return x

    def descaleX(_s, x):
        if _s.do_normalization:
            return _s.x_mean + (_s.x_std * x)
        else:
            return x

    def scaleY(_s, y, save=False):
        if _s.do_normalization:
            if save:
                _s.y_mean = np.mean(y)
                _s.y_std = np.std(y)
            return (y-_s.y_mean) / _s.y_std
        else:
            return y

    def descaleY(_s, y):
        if _s.do_normalization:
            return _s.y_mean + (_s.y_std * y)
        else:
            return y
