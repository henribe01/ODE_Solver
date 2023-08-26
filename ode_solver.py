import os
import sys
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
                        help='Plot CPU time vs step size.')
    parser.add_argument('-o', '--output', type=str,
                        help='Output file name. If not '
                             'specified, will show plot.')
    parser.add_argument('-sc', '--scipy', action='store_true',
                        help='Use scipy\'s odeint method to solve the ODE as well.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    try:
        args = parse_args(sys.argv[1:])
    except argparse.ArgumentError as e:
        sys.stderr.write(f'{sys.argv[0]}: {e}\n')
        sys.exit(1)

    # Import ODE
    ode_eq = import_module(f'ODEs.{args.ode}').__dict__[args.ode]

    # Check if initial conditions are the same length as the order of the ODE
    if len(args.initial_conditions) != ode_eq.__code__.co_argcount - 1:
        sys.stderr.write(f'{sys.argv[0]}: Initial conditions must be the same '
                         f'length as the order of the ODE.\n')
        sys.exit(1)

    # Set up plots
    amount_of_plots = 1 + args.error + args.cpu_time
    fig, axes = plt.subplots(amount_of_plots, 1)
    ax1 = axes[0] if amount_of_plots > 1 else axes
    fig.tight_layout(pad=3.0)

    # Plot SciPy solution
    if args.scipy:
        t_values = np.linspace(0, args.t_end, int(args.t_end / args.step_size))
        scipy_solution = odeint(scipy_ode, args.initial_conditions, t_values)[:,
                         0]
        ax1.plot(t_values, scipy_solution, color='red', label='Scipy',
                 linestyle='--', zorder=0)

    # Calc SciPy solution with different step sizes for error plot
    step_sizes = np.linspace(0.001, 0.5, 1000)
    scipy_solutions = {}
    if args.error:
        for step_size in step_sizes:
            t_values = np.linspace(0, args.t_end, int(args.t_end / step_size))
            scipy_solutions[step_size] = odeint(scipy_ode,
                                                args.initial_conditions,
                                                t_values)[:, 0]
    methods = get_all_methods()

    # Set args.method to all methods if args.all is True
    if args.all:
        args.methods = list(methods.keys())

    for method_name in args.methods:
        print(method_name)
        solver = methods[method_name](ode_eq, args.initial_conditions,
                                      args.step_size)
        solver.solve(args.t_end)
        solver.time_plot(ax1)

        # Plot error
        if args.error:
            ax2 = axes[1] if amount_of_plots > 1 else axes
            errors = []
            for step_size in step_sizes:
                solver = methods[method_name](ode_eq, args.initial_conditions,
                                              step_size)
                solver.solve(args.t_end)
                errors.append(solver.get_mse(scipy_solutions[step_size]))
            ax2.plot(step_sizes, errors, label=method_name)

        # Plot CPU time
        if args.cpu_time:
            ax3 = axes[2] if amount_of_plots > 1 else axes
            cpu_times = []
            for step_size in step_sizes:
                solver = methods[method_name](ode_eq, args.initial_conditions,
                                              step_size)
                result = timeit.timeit(lambda: solver.solve(args.t_end),
                                       number=1)
                cpu_times.append(result)
            ax3.plot(step_sizes, cpu_times, label=method_name)

    # Plot labels
    ax1.set_title(
        f'{args.ode.replace("_", " ").title()} Solution vs Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Solution')
    ax1.legend()
    if args.error:
        ax2.set_title(f'Error vs Step Size')
        ax2.set_xlabel('Step Size')
        ax2.set_ylabel('Error')
        ax2.set_yscale('log')
        ax2.legend()
    if args.cpu_time:
        ax3.set_title(f'CPU Time vs Step Size')
        ax3.set_xlabel('Step Size')
        ax3.set_ylabel('CPU Time')
        ax3.legend()
        ax3.set_yscale('log')
    if args.output:
        plt.savefig(args.output, dpi=300)
    else:
        plt.show()
