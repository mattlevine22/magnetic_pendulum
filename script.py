import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from tqdm import tqdm
import argparse

from utilities import load_data, dump_data, dict_to_file
from odelibrary import MagneticPendulum, my_solve_ivp
from pdb import set_trace as bp

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='results', type=str) # base directory for output
parser.add_argument("--training_box_size", default=1, type=float)
parser.add_argument("--testing_box_size", default=0.5, type=float)
parser.add_argument("--n_per_axis", default=20, type=int)
parser.add_argument("--n_random_features", default=100, type=int)
parser.add_argument("--fac_A", default=2, type=float) # 4 seemed good in range [0.1 to 5]
parser.add_argument("--alpha", default=1e-5, type=float)
parser.add_argument("--n_test_ics", default=5, type=int)
FLAGS = parser.parse_args()

## basic parameters
def main(training_box_size=1, testing_box_size=0.5, n_per_axis=20, n_random_features=100, alpha=1e-9, n_test_ics=5, fac_A=1, base_dir = '.'):

    output_dir = os.path.join(base_dir, 'trainBoxSize-{}_testBoxSize-{}_nPerAxis-{}_nRFs-{}_alpha-{}_facA-{}'.format(training_box_size, testing_box_size, n_per_axis, n_random_features, alpha, fac_A))
    os.makedirs(output_dir, exist_ok=True)

    pkl_path = os.path.join(output_dir, 'model_info.pkl')
    try:
        data = load_data(pkl_path)
        print('Successfully loaded model data!\n')
        print('Training MSE:', data['ode'].train_mse)
        print('Training R^2:', data['ode'].train_r2)
    except:
        print('Could not load existing data. Generating data and features...\n')
        ode = MagneticPendulum()
        u = ode.make_grid(box_size=training_box_size, n_per_axis=n_per_axis)
        udot = ode.rhs(u, t=0)

        ode.train_random_features(n_features=n_random_features, u_input=u, u_output=udot, alpha=alpha, fac_A=fac_A)

        ## save the data
        data = {'u': u,
                'udot': udot,
                'ode': ode,
                'fac_A': fac_A}
        dump_data(data, pkl_path)
        regr_data = {'train_mse':ode.train_mse, 'train_r2': ode.train_r2, 'coef_mean': ode.coef_mean, 'coef_std': ode.coef_std}
        dict_to_file(regr_data, os.path.splitext(pkl_path)[0]+'.txt')

    # make plots
    # Test trajectory predictions for a new ICs
    T = 100
    dt = 0.01
    t_eval = np.arange(start=0, stop=T+dt, step=dt)
    t_span = [t_eval[0], t_eval[-1]]
    settings = {'method': 'DOP853', 'rtol':1e-10, 'atol':1e-10}
    for i in range(n_test_ics):
        ic = data['ode'].get_ic(testing_box_size) # choose the size of initial box to draw random IC from.
        u_true = my_solve_ivp(ic, data['ode'].rhs, t_eval, t_span, settings)
        u_approx = my_solve_ivp(ic, data['ode'].rhs_approx, t_eval, t_span, settings)
        print('Max u_true = ', np.max(u_true))
        print('Min u_true = ', np.min(u_true))

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(u_true[:,1], '-b',  label='x true')
        ax.plot(u_approx[:,1], ':b',  label='x approx')
        ax.plot(u_true[:,3], '-r',  label='y true')
        ax.plot(u_approx[:,3], ':r',  label='y approx')
        ax.legend()
        fig.savefig(os.path.join(output_dir, 'test_plot_{}.pdf'.format(i)), format='pdf')
        plt.close()


if __name__ == "__main__":
    main(**FLAGS.__dict__)
