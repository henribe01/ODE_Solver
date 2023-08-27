import os
import sys
import time
import timeit

import numpy as np
import argparse

from matplotlib import pyplot as plt
from scipy.integrate import odeint

from ODESolver.base_classes import *
from ODESolver.explicit_solvers import *
from ODESolver.explicit_rk import *
from ODESolver.adaptive_rk import *
import inspect

from importlib import import_module


def get_all_methods():
    # Get all BaseClasses
    base_classes = inspect.getmembers(import_module('ODESolver.base_classes'),
                                      inspect.isclass)
    # Get all subclasses of ODESolverBase -> all methods
    methods = dict()
    for base_name, base_class in base_classes:
        for method in base_class.__subclasses__():
            method_name = method.__name__
            if 'Base' not in method_name:
                methods[method_name] = method
    return methods


def scipy_ode(y: np.ndarray, t: float) -> np.ndarray:
    """
    Converts a high order ODE to a system of first order ODEs.
    :param t: The time value.
    :param y: The solution vector.
    :return: The derivative vector.
    """
    return np.array([y[i] for i in range(1, len(y))] + [
        ode_eq(*y, t)])


def parse_args(argv):
    parser = argparse.ArgumentParser(prog='ODE Solver',
                                     description='Solve ODEs using different methods and'
                                                 'compare them')
    parser.add_argument('ode', type=str, default='simple_pendulum',
                        help='ODE to solve. Must be a file in the ODEs folder.'
                             'The file must contain a function with the same name.\n'
                             "The function must take the form f(x, x', ..., x^{(n-1)}, t) \n and return the value of x^{(n)} \n",
                        choices=[file[:-3] for file in os.listdir('ODEs') if
                                 file.endswith('.py')])
    parser.add_argument('-i', '--initial_conditions', type=float, nargs='+',
                        help='Initial conditions for the ODE. '
                             'Must be the same length as the order of the ODE.',
                        required=True)
    parser.add_argument('-t', '--t_end', type=float, default=10,
                        help='Time to solve up to.')
    parser.add_argument('-m', '--methods', type=str, nargs='+',
                        default=['ForwardEuler', 'Heun', 'RK4'],
                        help='Methods to use for solving the ODE.',
                        choices=get_all_methods().keys())
    parser.add_argument('-s', '--step_size', type=float, default=0.01,
                        help='Step size to use for solving the ODE.')
    parser.add_argument('-a', '--all', action='store_true',
                        help='Plot all methods in one plot.')
    parser.add_argument('-e', '--error', action='store_true',
                        help='Plot error vs step size.')
    parser.add_argument('-c', '--cpu_time', action='store_true',
                        help='Measure CPU time vs step size.')
    parser.add_argument('-o', '--output', type=str,
                        help='Output file name. If not '
                             'specified, will show plot.')
    parser.add_argument('-sc', '--scipy', action='store_true',
                        help='Use scipy\'s odeint method to solve the ODE as well.')

    return parser.parse_args(argv)


def setup_plots(args):
    amount_of_plots = 1 + args.error + args.cpu_time
    fig, axs = plt.subplots(amount_of_plots, 1)
    fig.tight_layout(pad=3.0)
    if amount_of_plots == 1:
        axs = [axs]

    # Plot Settings
    ode_ax = axs[0]
    ode_ax.set_title(f'{args.ode.replace("_", " ").title()} Solution vs Time')
    ode_ax.set_xlabel('Time')
    ode_ax.set_ylabel('x(t)')
    ode_ax.grid(True)
    ode_ax.set_xlim(0, args.t_end)

    # Error Plot Settings
    if args.error:
        error_ax = axs[1]
        error_ax.set_title(f'Error vs Step Size')
        error_ax.set_xlabel('Step Size')
        error_ax.set_ylabel('Error')
        error_ax.grid(True, which='both')
        error_ax.set_yscale('log')
        error_ax.set_xscale('log')

    # CPU Time Plot Settings
    if args.cpu_time:
        cpu_ax = axs[1 + args.error]
        cpu_ax.set_title(f'CPU Time vs Step Size')
        cpu_ax.set_xlabel('Step Size')
        cpu_ax.set_ylabel('CPU Time')
        cpu_ax.grid(True, which='both')
        cpu_ax.set_yscale('log')

    return fig, axs


if __name__ == '__main__':
    # Parse arguments and handle wrong input
    try:
        args = parse_args(sys.argv[1:])
    except argparse.ArgumentError as e:
        sys.stderr.write(f'{sys.argv[0]}: {e}\n')
        sys.exit(1)

    # Import the ODE function
    ode_eq = import_module(f'ODEs.{args.ode}').__dict__[args.ode]

    # Check if the initial conditions are the same length as the order of the ODE
    if len(args.initial_conditions) != ode_eq.__code__.co_argcount - 1:
        sys.stderr.write(f'{sys.argv[0]}: Initial conditions must be '
                         f'the same length as the order of the ODE.\n')
        sys.exit(1)

    # Set up plots
    fig, axs = setup_plots(args)

    # Solve ODE with Scipy if args.scipy
    if args.scipy:
        t_values = np.linspace(0, args.t_end, 1000)
        scipy_sol = odeint(scipy_ode, args.initial_conditions, t_values)[:, 0]
        axs[0].plot(t_values, scipy_sol, label='Scipy', color='red',
                    linestyle='--', zorder=0)

    # Set args.methods to all methods if args.all is True
    if args.all:
        args.methods = list(get_all_methods().keys())

    # Get all available methods
    available_methods = get_all_methods()

    # Solve ODE with each method
    for method in args.methods:
        solver = available_methods[method](ode_eq, args.initial_conditions,
                                           args.step_size)
        solver.solve(args.t_end)
        solver.time_plot(axs[0])

        # Plot error if args.error is True
        if args.error:
            errors = np.array([])
            for step_size in np.logspace(-3, -1, 100):
                solver = available_methods[method](ode_eq,
                                                   args.initial_conditions,
                                                   step_size)
                # To save time, when args.error is True, only solve up to t=1
                solver.solve(1)
                errors = np.append(errors, solver.get_mse())
            axs[1].plot(np.logspace(-3, -1, 100), errors, label=method)

        if args.cpu_time:
            cpu_times = np.array([])
            for step_size in np.logspace(-3, -1, 100):
                solver = available_methods[method](ode_eq,
                                                   args.initial_conditions,
                                                   step_size)
                # To save time, when args.cpu_time is True, only solve up to t=1
                result = timeit.timeit(lambda: solver.solve(1), number=1)
                cpu_times = np.append(cpu_times, result)
            axs[1 + args.error].plot(np.logspace(-3, -1, 100), cpu_times,
                                     label=method)

    # Add legend and set the symbols to not be transparent
    leg = axs[0].legend(loc='best', ncol=(len(args.methods) + args.scipy) // 2,
                        fancybox=True, fontsize='small')
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    # Save or show plot
    if args.output:
        fig.savefig(args.output, dpi=300)
    else:
        plt.show()
