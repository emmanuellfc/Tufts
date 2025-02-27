
test  = QMSolver(dt=0.001, dx=0.1, n=200, steps=200)
test.create_sho_hamiltonian()
test.ic_gaussian()
sols = test.solve()

# Plot Solution
plt.figure()
plt.plot(test.x, np.real(sols[0]), label='Re(psi) @ t = t0')
plt.plot(test.x, np.imag(sols[199]), label='Re(psi) @ t = tf')
plt.legend()
plt.show()

