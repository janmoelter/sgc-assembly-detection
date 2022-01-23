import networkx
import random

import numpy

import sys

from math import exp, log, floor
from scipy.special import gammaln

#/* Program to calculate the number of communities in a network using the
# * method of Newman and Reinert, which calculates a posterior probability
# * by Monte Carlo simulation of the integrated likelihood of a
# * degree-corrected stochastic block model
# *
# * Written by Mark Newman  6 APR 2016
# */

def estimate(graph, **kwargs):

    if not 'K' in kwargs: kwargs['K'] = 40
    if not 'K0' in kwargs: kwargs['K0'] = 40
    if not 'MCsweeps' in kwargs: kwargs['MCsweeps'] = 10000
    if not 'seed' in kwargs: kwargs['seed'] = None
    if not 'verbose' in kwargs: kwargs['verbose'] = True

    if kwargs['K0'] > kwargs['K']: kwargs['K0'] = kwargs['K']

    #/* Program control */

    VERBOSE = kwargs['verbose']
    
    #/* Constants */

    K = kwargs['K']                                                     #// Maximum number of groups
    K0 = kwargs['K0']                                                   #
    MCSWEEPS = kwargs['MCsweeps']                                       #// Number of Monte Carlo sweeps
    SAMPLE = 10                                                         #// Interval at which to print out results, in sweeps

    #/* Globals */

    G = None                                                            #// Struct storing the network
    twom = 0                                                            #// Twice the number of edges
    p = 0.                                                              #// Average edge probability

    global k; k = 0                                                     #// Current value of k
    global g; g = None                                                  #// Group assignments
    global n; n = None                                                  #// Group sizes
    global m; m = None                                                  #// Edge counts

    global lnfact; lnfact = None                                        #// Look-up table of log-factorials
    global E; E = None                                                  #// Log probability


    __k__ = numpy.empty(MCSWEEPS, dtype=numpy.uint8)
    #__g__ = list()
    __E__ = numpy.empty(MCSWEEPS, dtype=numpy.float64)



    #// Make a lookup table of log-factorial values

    def maketable():
        global lnfact

        t = 0
        length = 0

        length = twom + G['nvertices'] + 1
        lnfact = numpy.empty(length)
        for t in range(length):
            lnfact[t] = gammaln(t + 1)

    #// Log-probability function

    def logp(n, m):
        r, s = 0, 0
        kappa = 0
        res = 0.

        for r in range(k):
            res += lnfact[n[r]]

            if n[r] > 0:
                kappa = 0
                for s in range(k):
                    kappa += m[r][s]
                res += kappa * log(n[r]) + lnfact[n[r] - 1] - lnfact[kappa + n[r] - 1]
                res += lnfact[m[r][r] // 2] - (m[r][r] // 2 + 1) * log(0.5 * p * n[r] * n[r] + 1)
                for s in range(r + 1, k):
                    res += lnfact[m[r][s]] - (m[r][s] + 1) * log(p * n[r] * n[s] + 1)
        
        return res
    
    #// Initial group assignment

    def initgroups():
        global g
        global n
        global m

        global k
        global E

        i, u, v = 0, 0, 0
        r = 0

        #// Make the initial group assignments at random

        g = numpy.empty(G['nvertices'], dtype=numpy.uint16)
        for u in range(G['nvertices']):
            g[u] = random.randrange(K0)

        #// Calculate the values of the n's

        n = numpy.zeros(K, dtype=numpy.uint16)
        for u in range(G['nvertices']):
            n[g[u]] += 1

        #// Calcalate the values of the m's

        m = numpy.zeros((K,K), dtype=numpy.uint16)
        for u in range(G['nvertices']):
            for i in range(G['vertex'][u]['degree']):
                v = G['vertex'][u]['edge'][i]['target']
                m[g[u]][g[v]] += 1

        #// Initialize k and the log-probability

        k = K0
        E = logp(n, m)

    #// Function to update value of k

    def changek():
        global g
        global n
        global m

        global k

        r, s, u = 0, 0, 0
        kp = 0
        empty = 0
        # map = numpy.empty(K)
        sum = 0

        #// With probability 0.5, decrease k, otherwise increase it

        if random.random() < 0.5:

            #// Count the number of empty groups

            for r in range(k):
                if n[r] == 0:
                    empty += 1

            #// If there are any empty groups, remove one of them, or otherwise do nothing
            
            if empty > 0:

                #// If there is more than one empty group, choose at random which one to remove

                while True:
                    r = random.randrange(k)
                    if not n[r] > 0: break


                #// Decrease k by 1

                k -= 1

                #// Update the group labels

                for u in range(G['nvertices']):
                    if g[u] == k:
                        g[u] = r

                #// Update n_r

                n[r] = n[k]

                #// Update m_rs

                for s in range(k):
                    if r == s:
                        m[r][r] = m[k][k]
                    else:
                        m[r][s] = m[k][s]
                        m[s][r] = m[s][k]
        
        else:

            #// With probability k/(n+k) increase k by 1, adding an empty group

            if (G['nvertices'] + k) * random.random() < k:
                if k < K:
                    n[k] = 0
                    for r in range(k + 1):
                        m[k][r] = m[r][k] = 0
                    k += 1

    #// Function to update n and m for a proposed move

    def nmupdate(r, s, d):
        global n
        global m

        t = 0

        n[r] -= 1
        n[s] += 1
        for t in range(k):
            m[r][t] -= d[t]
            m[t][r] -= d[t]
            m[s][t] += d[t]
            m[t][s] += d[t]

    # Function that does one MCMC sweep (i.e., n individual moves) using the heatbath algorithm

    def sweep():
        global g

        global E

        i, j, u, v = 0, 0, 0, 0
        r, s = 0, 0
        #temp = 0
        accept = 0
        d = numpy.empty(K)
        x, Z, sum = 0., 0., 0.
        newE = numpy.empty(K)
        boltzmann = numpy.empty(K)

        for i in range(G['nvertices']):

            #// Optionally, perform a k-changing move

            if (G['nvertices'] + 1) * random.random() < 1.0:
                changek()

            #// Choose a random node

            u = random.randrange(G['nvertices'])
            r = g[u]

            #// Find the number of edges this node has to each group

            for s in range(k):
                d[s] = 0
            for j in range(G['vertex'][u]['degree']):
                v = G['vertex'][u]['edge'][j]['target']
                d[g[v]] += 1
            
            #// Calculate the probabilities of moving it to each group in turn

            Z = 0.
            for s in range(k):
                if s == r:
                    newE[s] = E
                else:
                    nmupdate(r, s, d)
                    newE[s] = logp(n, m)
                    nmupdate(s, r, d)
                boltzmann[s] = exp(newE[s] - E)
                Z += boltzmann[s]
            
            #// Choose which move to make based on these probabilities

            x = Z * random.random()
            sum = 0.
            for s in range(k):
                sum += boltzmann[s]
                if sum > x:
                    break

            #// Make the move

            if s != r:
                g[u] = s
                nmupdate(r, s, d)
                E = newE[s]
                accept += 1
        
        return accept / G['nvertices']

    #====================================================================

    #/* main() */

    u, r, s = 0, 0, 0

    #// Initialize the random number generator from the system clock
    
    random.seed(kwargs['seed'])

    #// Read the network from stdin

    if VERBOSE:
        print('Reading network...', end='\n', file=sys.stderr)

    G = graph
    twom = 0
    for u in range(G['nvertices']):
        twom += G['vertex'][u]['degree']
    p = twom / ( G['nvertices'] * G['nvertices'] )

    if VERBOSE:
        print('Read network with {:d} nodes and {:d} edges'.format(G['nvertices'] , int(twom/2)), end='\n', file=sys.stderr)

    #// Make the lookup table

    maketable()

    #// Initialize the group assignment

    initgroups()

    #// Perform the Monte Carlo

    for s in range(MCSWEEPS):

        sweep()

        __k__[s] = k
        #__g__[s,:] = g
        __E__[s] = E

        if s % SAMPLE == 0:
            #print('{:d} {:d} {:f}'.format(s, k, E), end='\n', file=sys.stdout)
            if VERBOSE:
                print('Sweep {:d}...'.format(s), end='\r', file=sys.stderr)

    if VERBOSE:
        print('', end='\n', file=sys.stderr)

    return list(__k__), list(__E__)
