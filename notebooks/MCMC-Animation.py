import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import multivariate_normal, norm

# Fix the random seed for replicability.
np.random.seed(12345678)

from matplotlib import animation
import matplotlib

from ipywidgets import interact, interactive, widgets
from IPython.display import display, HTML

from pymc3.stats import autocorr

# Define the distributions
p = multivariate_normal([0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]])

# Fixed number of samples
samples = 1000

# Fix lag time to calculate ACF
max_lags = 20

# Generate data and plot contour
xlim = [-2.4, 2.4]
ylim = [-2.4, 2.4]
x = np.arange(xlim[0], xlim[1], 0.1)
y = np.arange(ylim[0], ylim[1], 0.1)

# Quickly hacked plotting code
fig = plt.figure(figsize=(10, 10))
i_width = (xlim[0], xlim[1])
s_width = (ylim[0], ylim[1])
samples_width = (0, samples)

# Plot the subplots
ax1 = fig.add_subplot(221, xlim=i_width, ylim=samples_width)
ax2 = fig.add_subplot(224, xlim=samples_width, ylim=s_width)
ax3 = fig.add_subplot(223, xlim=i_width, ylim=s_width,
                      xlabel=r'$X_0$',
                      ylabel=r'$X_1$')
ax4 = fig.add_subplot(222, xlim=[0, max_lags], ylim=[0.0, 1.0],
                     xlabel='lag',
                     ylabel='ACF')
fig.subplots_adjust(wspace=0.0, hspace=0.0)

# Change a bit on subplot 4, wacky hacky
pos1 = ax4.get_position() # get the original position 
pos2 = [pos1.x0 + 0.1, pos1.y0 + 0.1,  pos1.width - 0.1, pos1.height - 0.1]
ax4.set_position(pos2) # set a new position

# Add the contour
X, Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
CS = ax3.contour(X, Y, p.pdf(pos))

# Add the lines
line1, = ax1.plot([], [], lw=1) # Top plot for X0
line2, = ax2.plot([], [], lw=1) # Right blot for X1
line3, = ax3.plot([], [], 'o', lw=2, alpha=.1) # The dots on the contour
line4, = ax3.plot([], [], lw=1, alpha=.3) # The path on the contour
line5, = ax3.plot([], [], 'k', lw=1) # The vertical line from the contour to the top
line6, = ax3.plot([], [], 'k', lw=1) # The horizontal line from the contour to the right
line7, = ax4.plot([], [], 'r', lw=1, label='X0') # ACF for X0
line8, = ax4.plot([], [], 'b', lw=1, label='X1') # ACF for X1
ax4.legend(loc="upper right")
ax1.set_xticklabels([]) # Remove x label
ax2.set_yticklabels([]) # Remove y label
lines = [line1, line2, line3, line4, line5, line6]


def init():
    for line in lines:
        line.set_data([], [])
    return lines

def animate(i, Z, max_lags):
    line1.set_data(Z[:i,0][::-1], range(len(Z[:i,0])))
    line2.set_data(range(len(Z[:i,1])), Z[:i,1][::-1])
    line3.set_data(Z[:i,0], Z[:i,1])
    line4.set_data(Z[:i,0], Z[:i,1])
    line5.set_data([Z[i-1,0], Z[i-1,0]], [Z[i-1,1], s_width[1]])
    line6.set_data([Z[i-1,0], i_width[1]], [Z[i-1,1], Z[i-1,1]])
    
    # Calculate the ACF and plot on graph after some samples
    if i >= samples*0.10:
        lags = np.arange(1, max_lags)
        line7.set_data(lags, [autocorr(Z[:i,0], l) for l in lags])
        line8.set_data(lags, [autocorr(Z[:i,1], l) for l in lags])
        
    return lines

'''
Metropolis-Hastings Sampling
'''
# Initilise value 
x = np.random.uniform(0.0, 1.0, size=2)

# Generate the 2D matrix
Z = np.zeros((samples+1, 2))
Z[0,:] = x

# Run Metropolis Sampling
t = 0
while t < samples:
    # Increase iteration
    t = t+1
    
    # Generate proposal distribution and take a sample
    q = multivariate_normal(x, [[1.0, 0.0], [0.0, 1.0]])
    xstar = q.rvs()
    
    # Calculate correction factor
    qstar = multivariate_normal(xstar, [[1.0, 0.0], [0.0, 1.0]])
    qprev = multivariate_normal(x, [[1.0, 0.0], [0.0, 1.0]])
    c = qstar.pdf(x)/qprev.pdf(xstar)
    
    # Calculate the acceptance probability
    alpha = min(1.0, (p.pdf(xstar)/p.pdf(x))*c)
    
    # Draw the uniform and check for acceptance/rejection
    u = np.random.uniform(0.0, 1.0, size=1)
    if u<=alpha:
        x = xstar
        Z[t,:] = x
    else:
        Z[t,:] = x

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

anim = animation.FuncAnimation(fig, animate, init_func=init, fargs=(Z, max_lags),
                               frames=samples, interval=50, blit=True)

anim.save('Metropolis.mp4', writer=writer)
# HTML(anim.to_html5_video())

'''
Component-wise Metropolis-Hastings Sampling
'''
# Initilise value 
x = np.random.uniform(0.0, 1.0, size=2)

# Generate the 2D matrix
Zc = np.zeros((samples+1, 2))
Zc[0,:] = x

# Run Component-wise Metropolis Sampling
t = 0
while t < samples:
    # Increase iteration
    t = t+1
    
    # Loop over dimension
    for i in range(len(x)):
        # Copy xstar
        xstar = x.copy()
        
        # Generate proposal distribution and take a sample
        qc = norm(loc=x[i], scale=1.0)
        xs = qc.rvs()
    
        # Calculate correction factor
        qs = norm(loc=xs, scale=1.0)
        qp = norm(loc=x[i], scale=1.0)
        c = qs.pdf(x[i])/qp.pdf(xs)
        # The calculation here is included for the sake of completeness
        # In fact, N(x, 1) is symmetric, hence c=1 all the time
        
        # Replace the dimension to create the proposal
        xstar[i] = xs
    
        # Calculate the acceptance probability
        alpha = min(1.0, (p.pdf(xstar)/p.pdf(x))*c)
    
        # Draw the uniform and check for acceptance/rejection
        u = np.random.uniform(0.0, 1.0, size=1)
        if u<=alpha:
            x[i] = xstar[i]
    
    # Record the value
    Zc[t,:] = x


anim = animation.FuncAnimation(fig, animate, init_func=init, fargs=(Zc, max_lags),
                               frames=samples, interval=20, blit=True)
anim.save('ComponentMetropolis.mp4', writer=writer)
# HTML(anim.to_html5_video())

'''
The Gibbs Sampler
'''
# Generate the 2D matrix
Zg = np.zeros((samples+1, 2))
Zg[0,:] = np.random.uniform(0.0, 1.0, size=2)

# Mean and covariance
mu = [0.0, 0.0]
cov = [0.8, 0.8] # 12 and 21

# Run Gibbs Sampler
t = 0
while t < samples:
    # Increase iteration
    t = t+1
    
    # Draw the first dimension
    Zg[t,0] = np.random.normal( mu[0] + cov[1]*(Zg[t-1,1] - mu[1]), np.sqrt(1 - cov[1]**2))
    
    # Then the second
    Zg[t,1] = np.random.normal( mu[1] + cov[0]*(Zg[t,0] - mu[0]), np.sqrt(1 - cov[0]**2))


# In[ ]:

anim = animation.FuncAnimation(fig, animate, init_func=init, fargs=(Zg, max_lags),
                               frames=samples, interval=20, blit=True)
anim.save('Gibbs.mp4', writer=writer)
# HTML(anim.to_html5_video())

