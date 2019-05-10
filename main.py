import numpy as np
# import hub_lats as hub
# from pyscf import fci
import evolve
import hams
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.interpolate import interp1d


def iFT(A):
    """
    Inverse Fourier transform
    :param A:  1D numpy.array
    :return:
    """
    A = np.array(A)
    minus_one = (-1) ** np.arange(A.size)
    result = np.fft.ifft(minus_one * A)
    result *= minus_one
    result *= np.exp(1j * np.pi * A.size / 2)
    return result


def FT(A):
    """
    Fourier transform
    :param A:  1D numpy.array
    :return:
    """
    # test
    A = np.array(A)
    minus_one = (-1) ** np.arange(A.size)
    result = np.fft.fft(minus_one * A)
    result *= minus_one
    result *= np.exp(-1j * np.pi * A.size / 2)
    return result


def progress(total, current):
    if total < 10:
        print("Simulation Progress: " + str(int(round(100 * current / total))) + "%")
    elif current % (total / 10) == 0:
        print("Simulation Progress: " + str(int(round(100 * current / total))) + "%")
    return


nelec = (2, 0)
nx = 4
ny = 0
# U/t is 4.0
# U = 4.0 * 0.52
U = 3
t = 0.52
delta = 1.e-2
cycles = 10

prop = hams.system(nelec, nx, ny, U, t, delta, cycles)

# expectations here
neighbour = []
phi_original = []
J_field = []
phi_reconstruct = [0, 0]
boundary_1 = []
boundary_2 = []
two_body = []

# Set Ground State
psi = prop.get_gs()[1].astype(np.complex128)

r = ode(evolve.integrate_f).set_integrator('zvode', method='bdf')
r.set_initial_value(psi, 0).set_f_params(prop)
branch = 0
# using scaled time
while r.successful() and r.t < prop.cycles:
    r.integrate(r.t + delta)
    psi = r.y
    time = r.t
    # add to expectations
    progress(prop.n_time, int(time / delta))
    neighbour.append(evolve.nearest_neighbour(prop, psi))
    phi_original.append(prop.phi(time))
    two_body.append(evolve.two_body(prop, psi))
    J_field.append(evolve.current(prop, phi_original[-1], neighbour[-1]))
    phi, branch = evolve.phi_reconstruct(prop, J_field[-1], neighbour[-1], phi_reconstruct[-1], phi_reconstruct[-2],
                                         branch)
    phi_reconstruct.append(phi)
    boundary_1.append(evolve.boundary_term_1(prop, psi))
    boundary_2.append(evolve.boundary_term_2(prop, psi))
del phi_reconstruct[0:2]

# attempting with real time
# real_limit=prop.cycles/prop.field
# real_delta=delta/prop.field
# while r.successful() and r.t < real_limit:
#     r.integrate(r.t + real_delta)
#     psi = r.y
#     time = r.t
#     # add to expectations
#     progress(int(real_limit / real_delta), int(time / real_delta))
#     neighbour.append(evolve.nearest_neighbour(prop, psi))
#     phi_original.append(prop.phi(time))
#     J_field.append(evolve.current(prop, phi_original[-1], neighbour[-1]))
#     phi, branch = evolve.phi_reconstruct(prop, J_field[-1], neighbour[-1], phi_reconstruct[-1], phi_reconstruct[-2],
#                                          branch)
#     phi_reconstruct.append(phi)
#     boundary_1.append(evolve.boundary_term_1(prop, psi))
#     boundary_2.append(evolve.boundary_term_2(prop, psi))
# del phi_reconstruct[0:2]

# Plotting
print(len(J_field))
t = np.linspace(0.0, prop.cycles, len(J_field))
neighbour = np.array(neighbour)
J_field = np.array(J_field)
phi_original = np.array(phi_original)
phi_reconstruct = np.array(phi_reconstruct)

# Plot the current expectation
plt.plot(t, J_field.real, label='original system')
plt.legend()
plt.xlabel('Time [cycles]')
plt.ylabel('current expectation')
plt.show()

# Plot the original and reconstructed phi field
plt.plot(t, phi_original, label='original')
plt.plot(t, phi_reconstruct, label='Reconstructed', linestyle='dashed')
plt.plot(t, np.ones(len(t)) * np.pi / 2, color='red')
plt.plot(t, np.ones(len(t)) * -1 * np.pi / 2, color='red')
plt.legend()
plt.xlabel('Time [cycles]')
plt.ylabel('$\\phi$')
plt.show()

# Boundary term plots
print(prop.t)
plt.plot(t, boundary_1)
plt.xlabel('Time [cycles]')
plt.ylabel('First Boundary Term')
plt.show()

plt.plot(t, boundary_2)
plt.xlabel('Time [cycles]')
plt.ylabel('Second Boundary term')
plt.show()

# Comparing current gradient with and without the given expressions.
boundary_1 = np.array(boundary_1)
boundary_2 = np.array(boundary_2)
two_body = np.array(two_body)
extra = 2*np.real(np.exp(1j * phi_original)*two_body)

plt.plot(t,extra)
plt.show()

diff = phi_original - np.angle(neighbour)
J_grad = -2 * prop.a * prop.t * np.gradient(phi_original) * np.abs(neighbour) * np.cos(diff)
# term_2 = prop.a * prop.t * prop.t * (
#         np.exp(-1j * 2 * phi_original) * boundary_2 + np.conjugate((np.exp(-1j * 2 * phi_original) * boundary_2)))
# term_1 = prop.a * prop.t * prop.t * (boundary_1)
plt.plot(t, J_grad + 2 * prop.a * prop.t * (
        np.gradient(np.angle(neighbour)) * np.abs(neighbour) * np.cos(diff) - np.gradient(
    np.abs(neighbour)) * np.sin(diff)), label='gradient calculated via expectations', linestyle='dashdot')
plt.plot(t, J_grad - prop.a * prop.t * prop.U * extra, linestyle='dashed',
         label='Gradient using commutators')
plt.plot(t, np.gradient(J_field.real), label='Numerical current gradient')
plt.xlabel('Time [cycles]')
plt.ylabel('Current Expectation Gradient')
plt.legend()
plt.show()
