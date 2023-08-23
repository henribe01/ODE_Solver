import os
import sys

import numpy as np
import argparse

from matplotlib import pyplot as plt
from scipy.integrate import odeint

from ODESolver.ode_solver import *
from importlib import import_module


# Features program should have:
# - Solve ODE and plot solution with one method and one step size
# - Solve ODE and plot solution with multiple methods and one step size
# - Plot Error vs step size for one method
# - Plot Error vs step size for multiple methods
# - Plot Time vs step size for one method
# - Plot Time vs step size for multiple methods

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
                        default=['Forwardeuler', 'Heun', 'Rk4'],
                        help='Methods to use for solving the ODE.',
                        choices=[method.__name__.capitalize() for method in
                                 ODESolverBase.__subclasses__()])
    parser.add_argument('-s', '--step_size', type=float, default=0.1,
                        help='Step size to use for solving the ODE.')
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

    if args.scipy:
        t_values = np.linspace(0, args.t_end, int(args.t_end / args.step_size))
        _ode = scipy_ode
        scipy_solution = odeint(_ode, args.initial_conditions, t_values)[:, 0]
        ax1.plot(t_values, scipy_solution, color='red', label='Scipy',
                 linestyle='--', zorder=0)

    methods = {method.__name__.capitalize(): method for method in
               ODESolverBase.__subclasses__()}

    for method_name in args.methods:
        solver = methods[method_name](ode_eq, args.initial_conditions,
                                      args.step_size)
        solver.solve(args.t_end)
        solver.time_plot(ax1)

        if args.error:
            # TODO: Implement error plot
            pass

        if args.cpu_time:
            # TODO: Implement CPU time plot
            pass

    ax1.set_title(
        f'{args.ode} with initial conditions {args.initial_conditions}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Solution')
    ax1.legend()
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()
