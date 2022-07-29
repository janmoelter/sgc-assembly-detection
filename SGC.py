import os
import sys

import time

import math
import random

import numpy as np
import scipy.io

import networkx as nx

import sklearn.cluster

from Modules.peakfinder import peakfinder
from Modules.estimate import estimate


MATLAB_INDEXING_COMPATIBILITY = True


class runtimer(object):
    
    def __init__(self):
        self.time = None
        
    def tic(self):
        self.time = time.time()
        return self.time
        
    def toc(self):
        if self.time is not None:
            __dt = time.time() - self.time
            self.time = None
            return __dt
        else:
            raise RuntimeError('runtimer.tic() must have been called before calling runtimer.toc().')

def printConsoleSection(s):
    """
    PRINTCONSOLESECTION(s)
    
       INPUT:
       input_args [str]: section title

       EXAMPLE:
       printConsoleSection('Hello World!')

    """

    S_Z = 80;
    
    print('_' * S_Z, file=sys.stdout)
    print(':: {}'.format(s.upper()), file=sys.stdout)
    print('', file=sys.stdout)

def print_timeinterval(dT):
    """
    PRINT_TIMEINTERVAL(dT) Prints time-interval in optimal units.

       INPUT:
       dT [float]: time interval in seconds

    """

    def timevec(t):
    
        mult = np.array([24, 60, 60, 1]);
        _timevec = np.zeros(len(mult));
        
        for j in range(len(mult)):
            _timevec[j], t = divmod(t, mult[j:].prod());
        
        return _timevec

    
    dTx = timevec(dT);
    
    i = np.where(dTx > 0)[0][0]
    
    v = np.array([[1, 1/(24), 1/(24 * 60), 1/(24 * 60 * 60)], [0, 1, 1/(60), 1/(60 * 60)], [0, 0, 1, 1/(60)], [0, 0, 0, 1]]) @ dTx
    u = np.array(['d', 'h', 'min', 's'])
    
    return '{:.1f}{:s}'.format(v[i], u[i])

def cross_evaluation(f, X, Y=None):
    """
    CROSS_EVALUATION(f, X, Y=None) Returns the matrix f(x,y) for x in X and y
    in Y.

       INPUT:
       f [function]: two-argument function
       X [list]: data points
       Y [list]: data points, =X if not explicitly given

    """

    if Y is None:
        Y = X
        
    OUT = np.nan * np.ones((len(X), len(Y)));
    
    for ix in range(len(X)):
        for iy in range(len(Y)):
            OUT[ix,iy] = f(X[ix], Y[iy])
            
    return OUT

def iif(boolean_argument, true_output, false_output):
    """
    IIF(boolean_argument, true_output, false_output) Inline if-then-else
    function to use in anonymous functions.

    """

    return true_output if boolean_argument else false_output

def seed_devrandom(verbose=None):
    """
    SEED_DEVRANDOM(verbose=None) Seeds the internal random number generator
    from the /dev/urandom device (only on Unix systems).

       INPUT:
       verbose [bool]

       OUPUT:
       output_args [int]: 4 byte random seed

    """

    if verbose is None:
        verbose = True;

    #if os.name == 'posix':
    #    try:
    #        if verbose:
    #            print('Opening /dev/urandom for reading 4 byte random seed ...', file=sys.stdout);
    #        
    #        with open('/dev/urandom', 'rb') as dev_random_:
    #            seed_ = int.from_bytes(dev_random_.read(4), byteorder='little')
    #        
    #        if verbose:
    #            print(' 4 byte random seed is 0x{:08X}.'.format(seed_), file=sys.stdout);
    #        random.seed(seed_)
    #    except:
    #        if verbose:
    #            print(' Reading 4 byte random seed from /dev/urandom failed.', file=sys.stdout);
    #        random.seed(None);
    #    end
    #else:
    #    random.seed(None);
    
    seed_ = int.from_bytes(os.urandom(4), byteorder='little')
    random.seed(seed_)
    
    OUT = seed_
    
    if verbose:
        print('Internal random number generator was initialised with seed 0x{:08X}.'.format(seed_), file=sys.stdout);
        print('', file=sys.stdout);
    
    return OUT


def findSignificantDF_FCoactivity(dF_F):
    """
    FINDSIGNIFICANTDF_FCOACTIVITY(dF_F)

       PARAMETERS:
       dF_F [TxN ndarray]: dF/F-signal for N units in T time steps

    """

    dF_F = dF_F[np.logical_not(np.any(np.isnan(dF_F), axis=1)),:];

    ## PREPROCESS THE dF/F-signal
    
    CONST = {}

    CONST['STD_SIG_THRESHOLD'] = 2;
    # \_ CONST['STD_SIG_THRESHOLD']: dF/F-signal significance level in standard
    # deviation from the mean

    print('significance threshold: {}STD'.format(CONST['STD_SIG_THRESHOLD']));

    T, N = dF_F.shape;

    mean_dF_F = dF_F.mean(axis=0, keepdims=True);
    # \_ mean_dF_F: mean dF/F-signal for N units
    std_dF_F = dF_F.std(axis=0, ddof=1, keepdims=True);
    # \_ std_dF_F: standard deviation in dF/F-signal for N units


    sig_dF_F_activity = (dF_F > mean_dF_F + CONST['STD_SIG_THRESHOLD'] * std_dF_F).astype('float');
    # \_ sig_dF_F_mask: binary mask of significant dF/F-signal for N units in T time steps


    ## FIND SIGNIFICANT PEAKS IN THE THE COACTIVITY

    CONST['SHUFFLE_ROUNDS'] = 1000;
    # \_ CONST['SHUFFLE_ROUNDS']: rounds of shuffling for coactivity null model
    CONST['SIGNIFICANCE_P'] = 0.05;
    # \_ CONST['SIGNIFICANCE_P']: significance level for the dF/F-coactivity

    sig_dF_F_coactivity_threshold, _ = findSignificantCoactivity(sig_dF_F_activity, **{'shuffle_rounds': CONST['SHUFFLE_ROUNDS'], 'significance_p': CONST['SIGNIFICANCE_P']});
    # \_ sig_dF_F_coactivity_threshold: significance threshold for the
    # dF/F-coactivity

    normalised_sig_dF_F_coactivity = sig_dF_F_activity.sum(axis=1) / sig_dF_F_activity.sum(axis=1).max();
    normalised_sig_dF_F_coactivity_threshold = sig_dF_F_coactivity_threshold / sig_dF_F_activity.sum(axis=1).max();
    # \_ normalised_sig_dF_F_coactivity: normalised dF/F-coactivity
    # \_ normalised_sig_dF_F_coactivity_threshold: normalised significance
    # threshold for the normalised dF/F-coactivity

    sig_dF_F_coactivity_peaks, _ = peakfinder(normalised_sig_dF_F_coactivity, 0.05, normalised_sig_dF_F_coactivity_threshold, 1, True, False);
    # \_ sig_dF_F_coactivity_peaks: peak times of the
    # dF/F-coactivity

    ## END

    return sig_dF_F_activity, sig_dF_F_coactivity_threshold, sig_dF_F_coactivity_peaks

def findSignificantCoactivity(X, shuffle_rounds, significance_p=0.05):
    """
    FINDSIGNIFICANTCOACTIVITY(X, shuffle_rounds, significance_p=0.05)

    """

    T, _ = X.shape;
    
    def shuffle(A, axis):
        _A = A.copy()
        np.apply_along_axis(np.random.shuffle, axis=axis, arr=_A)
        return _A
    # \_ shuffle(X, d): shuffle X along its d-th dimension, i.e. every slice
    # of X along its d-th dimension is shuffled independently
    # in particular, np.allclose(shuffle(X, d).sum(axis=d), X.sum(axis=d)) holds
    # true
    
    shuff_coactivity = np.zeros((shuffle_rounds, T));
    # \_ shuff_coactivity [opts.shuffle_rounds x T matrix]:
    
    for s in range(shuffle_rounds):
        shuff_coactivity[s,:] = shuffle(X, 0).sum(axis=1);
    
    sig_coactivity = np.percentile(shuff_coactivity.flatten(), q=(1 - significance_p) * 100);
    # \_ sig_coactivity: significant coactivity threshold estimated as the
    # (1-opts.significance_p)-percentile from shuffled coactiviy levels
    
    return sig_coactivity, shuff_coactivity

def findAssemblyPatterns(activityPatterns):
    """
    FINDASSEMBLYPATTERNS(activityPatterns)

       INPUT:
       activityPatterns [list]: list of binary activity patterns

       EXAMPLE:
       ---

    """

    OUT = {}

    OUT['activityPatterns'] = activityPatterns;

    ## > BUILD SIMILARITY GRAPH
    printConsoleSection('BUILD SIMILARITY GRAPH');
    
    print('{:d} activity patterns'.format(len(activityPatterns)), file=sys.stdout);
    print('', file=sys.stdout);
    
    if len(activityPatterns) == 0:
        return;
    
    patternSimilarityGraph = buildPatternSimilarityGraph(activityPatterns);
    #OUT['patternSimilarityGraph'] = patternSimilarityGraph;
    
    
    ## > ANALYSE COMMUNITY STRUCTURE IN SIMILARITY GRAPH
    printConsoleSection('ANALYSE COMMUNITY STRUCTURE IN SIMILARITY GRAPH');
    
    N_ITERATIONS = 5;
    N_MONTECARLOSTEPS = 50000;
        
    patternSimilarityAnalysis = analyseGraphCommunityStructure(patternSimilarityGraph, {'Iterations': N_ITERATIONS, 'MonteCarloSteps': N_MONTECARLOSTEPS, 'initialK': None});
    OUT['patternSimilarityAnalysis'] = patternSimilarityAnalysis;
    
    
    ## > INFER ASSEMBLY PATTERNS
    printConsoleSection('INFER ASSEMBLY PATTERNS');
    
    assemblyPatterns, iAssemblyPatterns = inferAssemblyPatterns(activityPatterns, patternSimilarityAnalysis);
    
    OUT['assemblyActivityPatterns'] = assemblyPatterns;
    OUT['assemblyIActivityPatterns'] = iAssemblyPatterns;
    
    print('   {:d} assembly patterns'.format(len(assemblyPatterns)), file=sys.stdout);
    print('', file=sys.stdout);
    
    return OUT

def inferAssemblyPatterns(activityPatterns, patternSimilarityAnalysis):
    """
    INFERASSEMBLYPATTERNS(activityPatterns, patternSimilarityAnalysis)

       INPUT:
       activityPatterns [list]: list of binary activity patterns
       patternSimilarityAnalysis [dict]: pattern similarity analysis results

    """

    MINIMUM_SIZE = 5;
    STD_DEVIATIONS = 1.5;

    ACTIVTY_THRESHOLD = 0.2;

    def discriminateSize():

        minSize = max(0, (lambda h : h.mean() - STD_DEVIATIONS * h.std(ddof=1))(np.array([(gAssignment == _).sum() for _ in np.unique(gAssignment) if not np.isnan(_)])));
        minSize = max(MINIMUM_SIZE, minSize);
    
        for r_ in [_ for _ in np.unique(gAssignment) if not np.isnan(_)]:
            if (gAssignment == r_).sum() < minSize:
                gAssignment[gAssignment == r_] = np.nan;


    cosineDistance = lambda x_1, x_2 : ( 1 - np.dot(x_1, x_2) / (np.linalg.norm(x_1) * np.linalg.norm(x_2)) );

    ## PERFORM SPECTRAL CLUSTERING ON THE SIMILARITY GRAPH

    gAssignment = spectralclustering(nx.from_numpy_array(patternSimilarityAnalysis['graph']), [patternSimilarityAnalysis['communityStructure']['count']], 'normalised')[0];

    ## DISREGARD COMMUNITIES CONSISTING OF TOO FEW ACTIVITY PATTERNS

    discriminateSize()

    ## DEFINE PRELIMINARY CORE ASSEMBLY PATTERNS

    prelimAssemblyPatterns = [None] * (int(np.nanmax(gAssignment)) + 1);

    for r in range(len(prelimAssemblyPatterns)):
        if np.any(gAssignment == r):
            prelimAssemblyPatterns[r] = meanActivityPattern(list(np.array(activityPatterns)[gAssignment == r,:]), ACTIVTY_THRESHOLD);
            prelimAssemblyPatterns[r] = (prelimAssemblyPatterns[r] != 0).astype('float');
        else:
            prelimAssemblyPatterns[r] = np.zeros(len(activityPatterns[0]));

    ## COMBINE SIMILAR PRELIMINARY CORE ASSEMBLY PATTERNS

    p = 2/3;

    prelimAssemblyPatternsSimilarity = cross_evaluation( lambda x, y: ( min( np.dot(x,y) / np.linalg.norm(y)**2 , np.dot(y,x) / np.linalg.norm(x)**2 ) > p ), prelimAssemblyPatterns).astype('bool');

    R, S = np.where(np.triu(prelimAssemblyPatternsSimilarity, 1))
    
    for i in range(np.triu(prelimAssemblyPatternsSimilarity, 1).sum()):
        gAssignment[gAssignment == S[i]] = R[i];
        
        # CHANGE THE REASSIGNMENT RECURSIVELY
        R[i + np.where(R[(i+1):] == S[i])[0]] = R[i];
    
    # RE-DEFINE PRELIMINARY CORE ASSEMBLY PATTERNS FROM THE CHANGED GROUP ASSIGNMENT

    for r in range(len(prelimAssemblyPatterns)):
        if np.any(gAssignment == r):
            prelimAssemblyPatterns[r] = meanActivityPattern(list(np.array(activityPatterns)[gAssignment == r,:]), ACTIVTY_THRESHOLD);
            prelimAssemblyPatterns[r] = (prelimAssemblyPatterns[r] != 0).astype('float');
        else:
            prelimAssemblyPatterns[r] = np.zeros(len(activityPatterns[0]));
    
    ## ASSIGN EVERY ACTIVITY TO A GROUP DEFINED BY A PRELIMINARY CORE ASSEMBLY PATTERN
    # IF THE PATTERNS DO NOT EXCEED A CERTAIN LEVEL OF SIMILARITY THEY WILL BE DISREGARDED
    
    p = 1/2;
    
    for j in range(len(gAssignment)):
        x = activityPatterns[j];
        r = np.argmin(np.array([cosineDistance(a,x) if not np.allclose(a, 0) else np.inf for a in prelimAssemblyPatterns]))
                   
        if r is not None:
            gAssignment[j] = iif( (np.dot(prelimAssemblyPatterns[r], x) > p * np.linalg.norm(prelimAssemblyPatterns[r])**2) and (np.linalg.norm(x)**2 > p * np.linalg.norm(prelimAssemblyPatterns[r])**2), r, -1);
    
        else:
            gAssignment[j] = -1;

    ## DISREGARD COMMUNITIES CONSISTING OF TOO FEW ACTIVITY PATTERNS
    
    discriminateSize()
    
    ## DEFINE CORE ASSEMBLY PATTERNS
    
    assemblyPatterns = [None] * (int(np.nanmax(gAssignment))+1);
    iAssemblyPatterns = [None] * (int(np.nanmax(gAssignment))+1);
    
    for r in range(len(assemblyPatterns)):
        if np.any(gAssignment == r):
            iAssemblyPatterns[r] = np.where(gAssignment == r)[0];
            assemblyPatterns[r] = meanActivityPattern([activityPatterns[_] for _ in iAssemblyPatterns[r]], ACTIVTY_THRESHOLD);


    assemblyPatterns = [_ for _ in assemblyPatterns if _ is not None];
    iAssemblyPatterns = [_ for _ in iAssemblyPatterns if _ is not None];


    return assemblyPatterns, iAssemblyPatterns
                   
def refreshAssemblyPatterns(assembly_pattern_output):
    """
    REFRESHASSEMBLYPATTERNS(assembly_pattern_output)

       INPUT:
       input_args [dict]: findAssemblyPattern output

    """

    OUT = assembly_pattern_output;
    
    ## > RE-ANALYSE COMMUNITY STRUCTURE IN SIMILARITY GRAPH
    printConsoleSection('RE-ANALYSE COMMUNITY STRUCTURE IN SIMILARITY GRAPH');

    k, k_distribution, g = computeGraphCommunityStructureMarginals(OUT['patternSimilarityAnalysis']['communityStructure']['markovChainMonteCarloSamples']);

    OUT['patternSimilarityAnalysis']['communityStructure']['count'] = k;
    OUT['patternSimilarityAnalysis']['communityStructure']['countDistribution'] = k_distribution;
    OUT['patternSimilarityAnalysis']['communityStructure']['assignment'] = [np.where(g == r)[0] for r in np.unique(g)];


    ## > INFER ASSEMBLY PATTERNS
    printConsoleSection('INFER ASSEMBLY PATTERNS');
    
    assemblyPatterns, iAssemblyPatterns = inferAssemblyPatterns(OUT['activityPatterns'], OUT['patternSimilarityAnalysis']);
    
    OUT['assemblyActivityPatterns'] = assemblyPatterns;
    OUT['assemblyIActivityPatterns'] = iAssemblyPatterns;
    
    print('   {:d} assembly patterns'.format(len(assemblyPatterns)));
    print('');

    return OUT

def buildPatternSimilarityGraph(X):
    """
    BUILDPATTERNSIMILARITYGRAPH(X) Returns a k-nearest-neighbour graph
    according to the consine distance from the pattern data in X.

       INPUT:
       X [list]: list of binary activity patterns

    """

    def k_nearestneighbours(X, k, d):

        OUT = [None] * len(X);

        dist = cross_evaluation(d, X) + np.diag(np.nan * np.ones(len(X)));

        for i in range(len(X)):
            s = np.sort(dist[i,:]);

            OUT[i] = np.where(dist[i,:] <= s[k-1])[0];

        return OUT
    

    cosineDistance = lambda x,y : 1 - ( np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y)) );
    # miDistance = lambda x,y : 1 - 2 * np.dot(x,y) / ( x.sum() + x.sum() );

    for k in list(range(math.ceil(math.log(len(X))), len(X)+1)):
    
        knn = k_nearestneighbours(X, k, cosineDistance);
        
        # We select the k nearest neighbours, however in case of a tie, when there
        # are multiple data point with the same distance from a single node, we
        # include all of them:
        
        A = np.zeros((len(knn), len(knn)));
        for x in range(len(knn)):
            n = knn[x][knn[x] != x];
            
            A[x, n] = 1;
            A[n, x] = 1;
        
        OUT = nx.from_numpy_array(A);
        
        if nx.is_connected(OUT):
            break;

    return OUT

def spectralclustering(G, K, normalisation=None):
    """
    SPECTRALCLUSTERING(G, K, normalisation=None) Performs spectral clustering
    on the graph G into k clusters.

       INPUT:
       G [graph]: input graph
       K [list]: number of clusters
       normalisation [str]: (optional) spectral clustering normalisation
       ('unnormalised','symmetric','randomwalk'/'normalised')

    """

    def graphlaplacian(G, normalisation=None):

        if normalisation is None:
            normalisation = '';

        A = nx.adjacency_matrix(G).A;
        D = np.array([k for _, k in G.degree()]);

        J = np.eye(len(G));

        if normalisation == '' or normalisation == 'unnormalised':
            d = D;
            OUT = np.diag(d) - A;
        elif normalisation == 'symmetric':
            d = 1 / np.sqrt(D);
            OUT = J - np.diag(d) @ A @ np.diag(d);
        elif normalisation == 'randomwalk' or normalisation == 'normalised':
            d = 1 / D;
            OUT = J - np.diag(d) @ A;

        return OUT

    def assignmentvector(G, c):

        OUT = np.zeros(len(G));

        for j in range(len(c)):
            OUT[c[j]] = j;


        OUT = normalisePatternEnumeration(OUT);

        return OUT


    if normalisation is None:
        normalisation = '';
        
    optimisation = False;

    # =========================================================================

    z = 1;
    H = graphlaplacian(G, normalisation) + z * np.eye(len(G));
    # In H the spectrum was shifted by an amount z away from 0 in order for the
    # matrix to be non-singular so that the computation of eigenvectors tends
    # to be more stable.

    e_, U_ = np.linalg.eig(H)
    iex = np.argsort(e_)
    e_ = e_[iex]

    OUT = np.nan * np.ones((len(K), len(G)));

    for j in range(len(K)):
        k = int(K[j]);
        
        e = e_[:k];
        U = U_[:,iex[:k]];
        
        # =========================================================================
        
        if np.all(U == np.real(U)) and np.abs(H @ U - U @ np.diag(e)).max() < 1e-10:
            
            if normalisation in ['' , 'unnormalised']:
                pass
            elif normalisation == 'symmetric':
                U = U / np.linalg.norm(U, axis=1, keepdims=True)
            elif normalisation in ['randomwalk', 'normalised']:
                pass
            
            V = np.array(range(len(G)));

            idx = sklearn.cluster.KMeans(n_clusters=k, n_init=100, max_iter=300).fit(np.real(U)).labels_
            idx = normalisePatternEnumeration(idx);

            OUT_k = [None] * k;
            for r in range(k):
                OUT_k[r] = V[idx == r];
            
            
            if optimisation and normalisation in ['randomwalk' , 'normalised']:
                #OUT_k = ncutoptimisation(G, OUT_k);
                pass
            
            iO = np.argsort([len(_) for _ in OUT_k])[::-1];
            OUT_k = [OUT_k[_] for _ in iO];
            
            OUT[j,:] = assignmentvector(G, OUT_k);
            
        else:
            print(' ! Spectral clustering failed.', file=sys.stdout);
            print('! Spectral clustering failed.', file=sys.stderr);
            OUT[j,:] = np.zeros(len(G));
    

    return OUT

def analyseGraphCommunityStructure(graph , opts=None):
    """
    ANALYSEGRAPHCOMMUNITYSTRUCTURE(graph, opts=None) Obtains an estimate for the
    community structure of a given graph.

       These functions implement the approach described in
       M. E. J. Newman and G. Reinert. "Estimating the number of communities
       in a network". Phys. Rev. Lett. 117 (2016).

       INPUT:
       graph [graph]: undirected, unweigted graph
       opts [dict]: (optional) parameters of the algorithm
           .Iterations [int]: number of independent MCMC runs (default: 1)
           .MonteCarloSteps [int]: number of steps in every MCMC run (default: 10000)
           .RNGSeed [int]: seed for the random number generator (default: None)
           .initialK [int]: initial guess for the number of communities (default: None)

       OUTPUT:
       output_args [dict]: results
           .graph [graph]: undirected, unweigted graph (same as
           input)
           .communityStructure.count [int]: most likely number of
           communities
           .communityStructure.countDistribution [?x2 ndarray]: probability
           distribution for the number of communities
           .communityStructure.assignment [list]: proposed community
           structure given the most likely number of communities
           .communityStructure.markovChainMonteCarloSamples [list]: MCMC
           samples

       EXAMPLES:
       analyseGraphCommunityStructure(ZacharyKarateClub , {'Iterations': 1, 'MonteCarloSteps': 10000, 'initialK': 3})

    """

    if opts is None:
        opts = {};
    
    opts_ = {};
    for opt in opts.keys():
        if False:
            pass
        elif opt == 'MonteCarloSteps':
            opts_['MonteCarloSteps'] = opts['MonteCarloSteps']
        elif opt == 'RNGSeed':
            opts_['RNGSeed'] = opts['RNGSeed'];
        elif opt == 'initialK':
            opts_['initialK'] = opts['initialK'];
            
    
    if not 'Iterations' in opts.keys():
        opts['Iterations'] = 1;
    
    STD_OPTS = {'MonteCarloSteps': 10000, 'RNGSeed': None, 'maximalK': math.ceil(len(graph)/3), 'initialK': None, 'showBanner': False};
    for opt in STD_OPTS.keys():
        if not opt in opts_.keys():
            opts_[opt] = STD_OPTS[opt];

    
    if len(graph) > 0:
        print('Initialising graph community structure estimation ...', file=sys.stdout);
        
        print(' Number of vertices: {:d}'.format(graph.number_of_nodes()), file=sys.stdout);
        print(' Number of edges: {:d}'.format(graph.number_of_edges()), file=sys.stdout);
        
        print('', file=sys.stdout);
        
        if opts_['RNGSeed'] is None:
            opts_['RNGSeed'] = seed_devrandom();
        
        if opts['Iterations'] > 1:
            print('Running graph community structure estimation ({:d} iterations) ...'.format(opts['Iterations']), file=sys.stdout);
        else:
            print('Running graph community structure estimation ...', file=sys.stdout);
        
        print('', file=sys.stdout);
        
        # estimation_samples = estimateGraphCommunityStructure( graph , opts_ );
        #try:
        estimation_samples = [None] * opts['Iterations'];
        for i in range(opts['Iterations']):
            timer = runtimer();
            timer.tic();
            estimation_samples[i] = estimateGraphCommunityStructure(graph, opts_);
            
            if opts['Iterations'] > 1:
                print(' Iteration {:d} completed: {:s}'.format(i+1, print_timeinterval(timer.toc())), file=sys.stdout);
                #print(estimation_samples[i]['E'][-1])
        
        print('', file=sys.stdout);
        print('Graph community structure estimation completed.', file=sys.stdout);
        print('', file=sys.stdout);
        #except:
        #    print('', file=sys.stdout);
        #    print('Graph community structure estimation failed.', file=sys.stdout);
        #    print('', file=sys.stdout);
        #    
        #    return None;
        
        k, k_distribution, g = computeGraphCommunityStructureMarginals(estimation_samples);

        OUT = {
            'graph': None,
            'communityStructure': {
                'count': None,
                'countDistribution': None,
                'assignment': None,
                'markovChainMonteCarloSamples': None
            }

        }
        
        OUT['graph'] = nx.to_numpy_array(graph);
        OUT['communityStructure']['count'] = k;
        OUT['communityStructure']['countDistribution'] = k_distribution;
        OUT['communityStructure']['assignment'] = [np.where(g == r)[0] for r in np.unique(g)];
        
        OUT['communityStructure']['markovChainMonteCarloSamples'] = estimation_samples;
    
    return OUT

def estimateGraphCommunityStructure(graph, opts=None):
    """
    ESTIMATEGRAPHCOMMUNITYSTRUCTURE(graph, opts) Wrapper function for the
    'estimate' function to run the Marko-Chain-Monte-Carlo procedure for
    estimating community structure.

       INPUT:
       graph [graph]: undirected, unweigted graph
       opts [dict]: (optional) parameters of the algorithm
           .MonteCarloSteps [int]: number of steps in every MCMC run (default: 10000)
           .RNGSeed [int]: seed for the random number generator (default: None)
           .maximalK [int]: maximal number of communities (default: 40)
           .initialK [int]: initial guess for the number of communities (default: None)
           .showBanner [bool]: show banner (default: False)

       OUTPUT:
       output_args [recarray]: results
           .k [int]: number of communities
           .g [1x? ndarray]: community assignment
           .E [float]: log-likelihood

       EXAMPLES:
       estimateGraphCommunityStructure(ZacharyKarateClub, {})

    """

    if type(graph) is not nx.classes.graph.Graph:
        raise(TypeError('graph'));
        
    if opts is None:
        opts = {}
    
    opts_ = {};
    for opt in opts.keys():
        if False:
            pass
        elif opt == 'MonteCarloSteps':
            opts_['MCsweeps'] = opts['MonteCarloSteps'];
        elif opt == 'RNGSeed':
            opts_['seed'] = opts['RNGSeed'];
        elif opt == 'maximalK':
            opts_['K'] = opts['maximalK'];
        elif opt == 'initialK':
            opts_['K0'] = opts['initialK'];
        else:
            pass
    
    STD_OPTS = {'K': 40, 'K0': None, 'seed': None, 'MCsweeps': 10000, 'verbose': True};
    for opt in STD_OPTS.keys():
        if not opt in opts_.keys():
            opts_[opt] = STD_OPTS[opt];
    
    if opts_['K0'] is None:
        opts_['K0'] = random.randint(1, opts_['K']);
    else:
        opts_['K0'] = min(opts_['K'], opts_['K0']);
        
    
    if 'showBanner' in opts.keys():
        if opts['showBanner']:
            
            print('', file=sys.stdout);
            
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++', file=sys.stdout);
            print('+                                                                              +', file=sys.stdout);
            print('+           estimate   ||   Newman, Reinert 2016 ; Phys.Rev.Lett 117           +', file=sys.stdout);
            print('+                                                                              +', file=sys.stdout);
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++', file=sys.stdout);
            
            print('', file=sys.stdout);
    
    
    k, E = estimate(graph, **opts_);
    
    
    k_dtype = 'uint{:d}'.format(2 ** (2 + np.where(np.log2(k).max() <= 2**(2+np.array([1,2,3,4])))[0][0] + 1))

    return np.array([(_[0], np.array([], dtype=k_dtype), _[1]) for _ in zip(k, E)], dtype=[('k', k_dtype), ('g', 'object'), ('E', 'float32')])

def computeGraphCommunityStructureMarginals(estimate_output):
    """
    COMPUTEGRAPHCOMMUNITYSTRUCTUREMARGINALS(estimate_output) Computes the
    marginal distributions to estimate community structure of a graph given
    the results of (several rounds) of Markov-Chain-Monte-Carlo sampling to
    from the function ESTIMATEGRAPHCOMMUNITYSTRUCTURE.

       INPUT:
       estimate_output [list]: estimate-MCMC samples from
       ESTIMATEGRAPHCOMMUNITYSTRUCTURE.

       OUTPUT:
       k_ [int]: most likely number of communities
       k_distibution [?x2 ndarray]: probability distribution for the number of
       communities
       g_ [1x? ndarray]: proposed community structure given the most likely
       number of communities 

    """

    K = [];
    
    for i in range(len(estimate_output)):
        
        estimate_output_E = estimate_output[i]['E'];
        EQ_sample = estimate_output_E[math.floor(len(estimate_output_E)/2):];
        
        i_EQ = estimate_output_E > EQ_sample.mean() - 2 * np.abs(EQ_sample - EQ_sample.mean()).max();
        i_EQ[:np.where(np.invert(i_EQ))[0][-1]] = False;
        
        K_ = estimate_output[i]['k'];
        K = np.concatenate([K , K_[i_EQ]]);
    
    #
    
    H = np.arange(K.min(), K.max()+1)
    k_distibution = np.array([H, np.histogram(K, bins=np.concatenate([H, np.array([H[-1]+1])]), density=True)[0]]).T
    
    j = np.argmax(k_distibution[:,1]);
    k_ = k_distibution[j,0];
    
    #
    
    
    return k_ , k_distibution , np.array([])

def meanActivityPattern(activityPatterns, activityThreshold):
    """
    MEANACTIVITYPATTERN(activityPatterns, activityThreshold)
    
    """

    ## AVERAGE ACTIVITY PATTERNS
    OUT = np.array(activityPatterns).mean(axis=0)

    ##THRESHOLD THE AVERAGE PATTERN
    OUT = OUT * (OUT > activityThreshold);

    return OUT

def normalisePatternEnumeration(pattern):
    """
    NORMALISEPATTERNENUMERATION(pattern) Returns a enumeration of the input
    pattern Given the input pattern [2, 4, 3, 1, 1, 3, 4, 2] this function
    turns is into [0, 1, 2, 3, 3, 2, 1, 0] keeping the overall pattern
    unchanged but imposing a normalised enumeration.

       EXAMPLE:
       normalisePatternEnumeration(np.array([2, 4, 3, 1, 1, 3, 4, 2]))

    """

    OUT = np.nan * np.ones(pattern.shape)
    
    G, i = np.unique(pattern, return_index=True)
    G = G[np.argsort(i)]
    
    for r in range(len(G)):
        OUT[pattern == G[r]] = r
        
    return OUT


def load_CALCIUM_FLUORESCENCE_mat(filename):
    
    CALCIUM_FLUORESCENCE_mat = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)

    __CALCIUM_FLUORESCENCE_mat = {
        'calcium_fluorescence': {
            'F': None,
            'F0': None,
            'dF_F': None
        },
        'topology': None,
        'parameter': {
            'units': None,
            'dT_step': None,
            'time_steps': None,
            'assembly_configuration': None,
            'rate_range': None,
            'eventDuration': None,
            'eventFreq': None,
            'eventMult': None,
            'calcium_T1_2': None,
            'saturation_K': None,
            'noiseSTD': None
        },
        'meta_information': None
    }

    __CALCIUM_FLUORESCENCE_mat['calcium_fluorescence']['F'] = CALCIUM_FLUORESCENCE_mat['calcium_fluorescence'].F
    __CALCIUM_FLUORESCENCE_mat['calcium_fluorescence']['F0'] = CALCIUM_FLUORESCENCE_mat['calcium_fluorescence'].F0
    __CALCIUM_FLUORESCENCE_mat['calcium_fluorescence']['dF_F'] = CALCIUM_FLUORESCENCE_mat['calcium_fluorescence'].dF_F
    
    __CALCIUM_FLUORESCENCE_mat['topology'] = [_.astype('float') for _ in list(CALCIUM_FLUORESCENCE_mat['topology'])]
    
    __CALCIUM_FLUORESCENCE_mat['parameter']['units'] = CALCIUM_FLUORESCENCE_mat['parameter'].units
    __CALCIUM_FLUORESCENCE_mat['parameter']['dT_step'] = CALCIUM_FLUORESCENCE_mat['parameter'].dT_step
    __CALCIUM_FLUORESCENCE_mat['parameter']['time_steps'] = CALCIUM_FLUORESCENCE_mat['parameter'].time_steps
    if isinstance(CALCIUM_FLUORESCENCE_mat['parameter'].assembly_configuration, np.ndarray):
        __CALCIUM_FLUORESCENCE_mat['parameter']['assembly_configuration'] = list(CALCIUM_FLUORESCENCE_mat['parameter'].assembly_configuration)
    __CALCIUM_FLUORESCENCE_mat['parameter']['rate_range'] = CALCIUM_FLUORESCENCE_mat['parameter'].rate_range
    if isinstance(__CALCIUM_FLUORESCENCE_mat['parameter']['rate_range'], np.ndarray):
        __CALCIUM_FLUORESCENCE_mat['parameter']['rate_range'] = tuple(__CALCIUM_FLUORESCENCE_mat['parameter']['rate_range'])
    __CALCIUM_FLUORESCENCE_mat['parameter']['eventDuration'] = CALCIUM_FLUORESCENCE_mat['parameter'].eventDuration
    __CALCIUM_FLUORESCENCE_mat['parameter']['eventFreq'] = CALCIUM_FLUORESCENCE_mat['parameter'].eventFreq
    __CALCIUM_FLUORESCENCE_mat['parameter']['eventMult'] = CALCIUM_FLUORESCENCE_mat['parameter'].eventMult
    __CALCIUM_FLUORESCENCE_mat['parameter']['calcium_T1_2'] = CALCIUM_FLUORESCENCE_mat['parameter'].calcium_T1_2
    __CALCIUM_FLUORESCENCE_mat['parameter']['saturation_K'] = CALCIUM_FLUORESCENCE_mat['parameter'].saturation_K
    __CALCIUM_FLUORESCENCE_mat['parameter']['noiseSTD'] = CALCIUM_FLUORESCENCE_mat['parameter'].noiseSTD
    
    if 'meta_information' in CALCIUM_FLUORESCENCE_mat:
        __CALCIUM_FLUORESCENCE_mat['meta_information'] = {fieldname: fieldvalue for fieldname, fieldvalue in CALCIUM_FLUORESCENCE_mat['meta_information'].__dict__.items() if not fieldname == '_fieldnames'}


    # Ensure compatibility with MATLAB with regard to 1-based indexing
    if __CALCIUM_FLUORESCENCE_mat['parameter']['assembly_configuration'] is not None:
        __CALCIUM_FLUORESCENCE_mat['parameter']['assembly_configuration'] = list(map(lambda I: I-int(MATLAB_INDEXING_COMPATIBILITY), __CALCIUM_FLUORESCENCE_mat['parameter']['assembly_configuration']))

    # --------------------------------------------------------------------------
    
    return __CALCIUM_FLUORESCENCE_mat

def load_ACTIVITY_RASTER_mat(filename):
    
    ACTIVITY_RASTER_mat = scipy.io.loadmat(filename, struct_as_record=False)
    
    __ACTIVITY_RASTER_mat = {
        'activity_raster': None,
        'activity_raster_peak_threshold': None,
        'activity_raster_peaks': None
    }
    
    __ACTIVITY_RASTER_mat['activity_raster'] = ACTIVITY_RASTER_mat['activity_raster']
    __ACTIVITY_RASTER_mat['activity_raster_peak_threshold'] = ACTIVITY_RASTER_mat['activity_raster_peak_threshold'][0][0]
    __ACTIVITY_RASTER_mat['activity_raster_peaks'] = ACTIVITY_RASTER_mat['activity_raster_peaks']


    # Ensure compatibility with MATLAB with regard to 1-based indexing
    __ACTIVITY_RASTER_mat['activity_raster_peaks'] -= int(MATLAB_INDEXING_COMPATIBILITY)

    # --------------------------------------------------------------------------

    return __ACTIVITY_RASTER_mat

def save_ACTIVITY_RASTER_mat(filename, ACTIVITY_RASTER_mat):

    __ACTIVITY_RASTER_mat = ACTIVITY_RASTER_mat

    __ACTIVITY_RASTER_mat['activity_raster_peaks'] = __ACTIVITY_RASTER_mat['activity_raster_peaks'][:,np.newaxis]

    # Ensure compatibility with MATLAB with regard to 1-based indexing
    __ACTIVITY_RASTER_mat['activity_raster_peaks'] += int(MATLAB_INDEXING_COMPATIBILITY)

    # --------------------------------------------------------------------------

    scipy.io.savemat(filename, __ACTIVITY_RASTER_mat)

def load_SGC_ASSEMBLIES_mat(filename):
    
    SGC_ASSEMBLIES_mat = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    
    __SGC_ASSEMBLIES_mat = {
        'assembly_pattern_detection': {
            'activityPatterns': None,
            'patternSimilarityAnalysis': {
                'graph': None,
                'communityStructure': None
            },
            'assemblyActivityPatterns': None,
            'assemblyIActivityPatterns': None,
        },
        'assemblies': None
    }
    __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['activityPatterns'] = list(SGC_ASSEMBLIES_mat['assembly_pattern_detection'].activityPatterns)
    
    if type(SGC_ASSEMBLIES_mat['assembly_pattern_detection'].patternSimilarityAnalysis.graph) is scipy.io.matlab.mio5_params.MatlabOpaque:
        raise ValueError('`assembly_pattern_detection.patternSimilarityAnalysis.graph` is a MATLAB class object unknown to SciPy.')
    else:
        __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['patternSimilarityAnalysis']['graph'] = SGC_ASSEMBLIES_mat['assembly_pattern_detection'].patternSimilarityAnalysis.graph
    
    __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['patternSimilarityAnalysis']['communityStructure'] = {
        'count': SGC_ASSEMBLIES_mat['assembly_pattern_detection'].patternSimilarityAnalysis.communityStructure.count,
        'countDistribution': SGC_ASSEMBLIES_mat['assembly_pattern_detection'].patternSimilarityAnalysis.communityStructure.countDistribution,
        'assignment': list(SGC_ASSEMBLIES_mat['assembly_pattern_detection'].patternSimilarityAnalysis.communityStructure.assignment),
        'markovChainMonteCarloSamples': [np.array([(_.k, _.g, _.E) for _ in _], dtype=[('k', 'uint8'), ('g', 'object'), ('E', 'float32')]) for _ in SGC_ASSEMBLIES_mat['assembly_pattern_detection'].patternSimilarityAnalysis.communityStructure.markovChainMonteCarloSamples]
    }
    __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['assemblyActivityPatterns'] = list(SGC_ASSEMBLIES_mat['assembly_pattern_detection'].assemblyActivityPatterns)
    __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['assemblyIActivityPatterns'] = list(SGC_ASSEMBLIES_mat['assembly_pattern_detection'].assemblyIActivityPatterns)
    __SGC_ASSEMBLIES_mat['assemblies'] = list(SGC_ASSEMBLIES_mat['assemblies'])


    # Ensure compatibility with MATLAB with regard to 1-based indexing
    __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['patternSimilarityAnalysis']['communityStructure']['assignment'] = list(map(lambda I: I-int(MATLAB_INDEXING_COMPATIBILITY), __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['patternSimilarityAnalysis']['communityStructure']['assignment']))
    
    for r in range(len(__SGC_ASSEMBLIES_mat['assembly_pattern_detection']['patternSimilarityAnalysis']['communityStructure']['markovChainMonteCarloSamples'])):
        __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['patternSimilarityAnalysis']['communityStructure']['markovChainMonteCarloSamples'][r]['E'] -= int(MATLAB_INDEXING_COMPATIBILITY)

    __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['assemblyIActivityPatterns'] = list(map(lambda I: I-int(MATLAB_INDEXING_COMPATIBILITY), __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['assemblyIActivityPatterns']))
    __SGC_ASSEMBLIES_mat['assemblies'] = list(map(lambda I: I-int(MATLAB_INDEXING_COMPATIBILITY), __SGC_ASSEMBLIES_mat['assemblies']))

    # --------------------------------------------------------------------------
    
    return __SGC_ASSEMBLIES_mat

def save_SGC_ASSEMBLIES_mat(filename, SGC_ASSEMBLIES_mat):

    __SGC_ASSEMBLIES_mat = SGC_ASSEMBLIES_mat

    # Ensure compatibility with MATLAB with regard to 1-based indexing
    __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['patternSimilarityAnalysis']['communityStructure']['assignment'] = list(map(lambda I: I+int(MATLAB_INDEXING_COMPATIBILITY), __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['patternSimilarityAnalysis']['communityStructure']['assignment']))
    
    for r in range(len(__SGC_ASSEMBLIES_mat['assembly_pattern_detection']['patternSimilarityAnalysis']['communityStructure']['markovChainMonteCarloSamples'])):
        __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['patternSimilarityAnalysis']['communityStructure']['markovChainMonteCarloSamples'][r]['E'] += int(MATLAB_INDEXING_COMPATIBILITY)

    __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['assemblyIActivityPatterns'] = list(map(lambda I: I+int(MATLAB_INDEXING_COMPATIBILITY), __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['assemblyIActivityPatterns']))
    __SGC_ASSEMBLIES_mat['assemblies'] = list(map(lambda I: I+int(MATLAB_INDEXING_COMPATIBILITY), __SGC_ASSEMBLIES_mat['assemblies']))

    # --------------------------------------------------------------------------

    def as_objectarray(list):
        _array = np.array([None] * len(list), dtype=object)
        for i, x in enumerate(list):
            _array[i] = x
    
        return _array

    __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['activityPatterns'] = as_objectarray(__SGC_ASSEMBLIES_mat['assembly_pattern_detection']['activityPatterns'])[:,np.newaxis]

    __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['patternSimilarityAnalysis']['communityStructure']['assignment'] = as_objectarray(__SGC_ASSEMBLIES_mat['assembly_pattern_detection']['patternSimilarityAnalysis']['communityStructure']['assignment'])[np.newaxis,:]
    __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['patternSimilarityAnalysis']['communityStructure']['markovChainMonteCarloSamples'] = as_objectarray(__SGC_ASSEMBLIES_mat['assembly_pattern_detection']['patternSimilarityAnalysis']['communityStructure']['markovChainMonteCarloSamples'])[:,np.newaxis]
    
    __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['assemblyActivityPatterns'] = as_objectarray(__SGC_ASSEMBLIES_mat['assembly_pattern_detection']['assemblyActivityPatterns'])[:,np.newaxis]
    __SGC_ASSEMBLIES_mat['assembly_pattern_detection']['assemblyIActivityPatterns'] = as_objectarray(__SGC_ASSEMBLIES_mat['assembly_pattern_detection']['assemblyIActivityPatterns'])[:,np.newaxis]
    
    __SGC_ASSEMBLIES_mat['assemblies'] = as_objectarray(__SGC_ASSEMBLIES_mat['assemblies'])[:,np.newaxis]

    scipy.io.savemat(filename, __SGC_ASSEMBLIES_mat)


def CALCIUM_FLUORESCENCE_PROCESSING(CALCIUM_FLUORESCENCE_file):

    OUT = {}

    if os.path.isfile(CALCIUM_FLUORESCENCE_file):

        printConsoleSection( 'PROCESS CALCIUM FLUORESCENCE' );

        CALCIUM_FLUORESCENCE_mat = load_CALCIUM_FLUORESCENCE_mat(CALCIUM_FLUORESCENCE_file);

        print('Prepare *_ACTIVITY-RASTER.MAT file for  similarity-graph-clustering ...', file=sys.stdout);

        if not os.path.isfile(CALCIUM_FLUORESCENCE_file.replace('_CALCIUM-FLUORESCENCE', '_ACTIVITY-RASTER')):

            timer = runtimer();
            timer.tic();
            sig_dF_F_activity, sig_dF_F_coactivity_threshold, sig_dF_F_coactivity_peaks = findSignificantDF_FCoactivity(CALCIUM_FLUORESCENCE_mat['calcium_fluorescence']['dF_F']);

            OUT['activity_raster'] = sig_dF_F_activity;
            OUT['activity_raster_peak_threshold'] = sig_dF_F_coactivity_threshold;
            OUT['activity_raster_peaks'] = sig_dF_F_coactivity_peaks;
            

            directory, name = os.path.split(CALCIUM_FLUORESCENCE_file);
            name, _ = os.path.splitext(name);

            OUTPUT_PATH = os.path.join(directory, name.replace('_CALCIUM-FLUORESCENCE', '_ACTIVITY-RASTER') + '.mat');
            save_ACTIVITY_RASTER_mat(OUTPUT_PATH, OUT);
            #os.chmod(OUTPUT_PATH, os.stat(OUTPUT_PATH).st_mode | stat.S_IWGRP);

            timer.toc();

        print('', file=sys.stdout);

        ## >

        print('>> END PROGRAM', file=sys.stdout);
        print('>> END PROGRAM', file=sys.stderr);

    else:

        print('>> END PROGRAM', file=sys.stdout);
        print('>> END PROGRAM', file=sys.stderr);
        
def SGC_ASSEMBLY_DETECTION(ACTIVITY_RASTER_file):
    
    OUT = {}

    if os.path.isfile(ACTIVITY_RASTER_file):
        
        if not os.path.isfile(ACTIVITY_RASTER_file.replace('_ACTIVITY-RASTER', '_SGC-ASSEMBLIES')):
            
            printConsoleSection('INITIALISE ASSEMBLY PATTERN DETECTION');
            
            ACTIVITY_RASTER_mat = load_ACTIVITY_RASTER_mat(ACTIVITY_RASTER_file)
            
            activity_patters = list(ACTIVITY_RASTER_mat['activity_raster'][ACTIVITY_RASTER_mat['activity_raster_peaks'].flatten(),:])
            
            printConsoleSection('RUN ASSEMBLY PATTERN DETECTION');
            
            OUT['assembly_pattern_detection'] = findAssemblyPatterns(activity_patters);
            
        else:
            
            printConsoleSection('INITIALISE ASSEMBLY PATTERN DETECTION');
            
            OUT = load_SGC_ASSEMBLIES_mat(ACTIVITY_RASTER_file.replace('_ACTIVITY-RASTER', '_SGC-ASSEMBLIES'));
            
            printConsoleSection('REDO ASSEMBLY PATTERN DETECTION');
            
            OUT['assembly_pattern_detection'] = refreshAssemblyPatterns(OUT['assembly_pattern_detection']);
            
      
        OUT['assemblies'] = [np.where(_ > 0)[0] for _ in OUT['assembly_pattern_detection']['assemblyActivityPatterns']];
        
        
        printConsoleSection('SAVE RESULTS');
    
        directory, name = os.path.split(ACTIVITY_RASTER_file)
        name, _ = os.path.splitext(name)

        OUTPUT_PATH = os.path.join(directory, name.replace('_ACTIVITY-RASTER', '_SGC-ASSEMBLIES') + '.mat')
        save_SGC_ASSEMBLIES_mat(OUTPUT_PATH, OUT);
        #os.chmod(OUTPUT_PATH, os.stat(OUTPUT_PATH).st_mode | stat.S_IWGRP);
    
        print('>> END PROGRAM', file=sys.stdout);
        print('>> END PROGRAM', file=sys.stderr);

        return OUT;
        
    else:
        OUT = None;
        
        print('>> END PROGRAM', file=sys.stdout);
        print('>> END PROGRAM', file=sys.stderr);



if __name__ == "__main__":
    # ********************************************************************************
    # Argument parsing
    #

    import argparse
    import traceback


    __parser = argparse.ArgumentParser(
        description='Detection of neural assemblies the Similarity-Graph-Clustering (SGC) algorithm.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    __subparsers = __parser.add_subparsers(title='subcommands',dest='subcommand', required=True, help='')

    __subparser = dict()
    __subparser['preprocessing'] = __subparsers.add_parser('preprocessing',
        description='Transforms the calcium fluorescence signal into a raster of binary activity patterns.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    __subparser['detection'] = __subparsers.add_parser('detection',
        description='Performs the assembly detection using Similarity-Graph-Clustering (SGC) algorithm.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # *** 'preprocessing' ***

    __subparser['preprocessing'].add_argument('input-file', type=str, metavar='<file name>', help='`*_CALCIUM-FLUORESCENCE.mat`-file.')

    # *** 'detection' ***

    __subparser['detection'].add_argument('input-file', type=str, metavar='<file name>', help='`*_ACTIVITY-RASTER.mat`-file.')

    kwargs = vars(__parser.parse_args())
    
    # ********************************************************************************
    # Preprocess arguments

    __parser_subcommand = kwargs.pop('subcommand')

    if __parser_subcommand == 'preprocessing':
        pass

    if __parser_subcommand == 'detection':
        pass

    # ********************************************************************************
    # Execute main function
    
    try:
        if __parser_subcommand == 'detection':
            SGC_ASSEMBLY_DETECTION(kwargs['input-file'])
            #print(kwargs['input-file'])
        elif __parser_subcommand == 'preprocessing':
            CALCIUM_FLUORESCENCE_PROCESSING(kwargs['input-file'])
            #print(kwargs['input-file'])
            
    except:
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
