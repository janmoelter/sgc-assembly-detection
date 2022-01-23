import warnings

import networkx as nx


def estimate(graph, K=40, K0=40, MCsweeps=10000, seed=None, verbose=False):

    """
    Function to calculate the number of communities in a network using the method of Newman and Reinert, which calculates a posterior probability by Monte Carlo simulation of the integrated likelihood of a degree-corrected stochastic block model.

    This function implements the approach described in M. E. J. Newman and G. Reinert. "Estimating the number of communities in a network". Phys. Rev. Lett. 117 (2016).
    Whenever possible this function aims to use the original by M. E. J. Newman written in C on 6. April 2016 adapted to be integrated as a Python extension. If this fails, it will fall back to an essentially 1:1 translation of the original function in Python, for which the author does not claim any originality.
    """
    
    ALWAYS_USE_NATIVE_IMPLEMENTATION = False;
    

    assert type(graph) is nx.classes.graph.Graph, 'Argument `graph` is not of type networkx.classes.graph.Graph.'


    def graph2NETWORK(graph):

        assert type(graph) is nx.classes.graph.Graph, 'Argument `graph` is not of type networkx.classes.graph.Graph.'

        NETWORK = dict()
        NETWORK['nvertices'] = graph.number_of_nodes()
        NETWORK['vertex'] = list()

        __nodes = list(graph.nodes())
        for v in graph.nodes():

            __id = __nodes.index(v)
            __degree = graph.degree(v)
            __label = str(v)

            __edge = list()

            for u in graph.neighbors(v):

                __target = __nodes.index(u)
                __weight = 1

                __edge.append({'target': __target, 'weight': __weight})

            NETWORK['vertex'].append({'id': __id, 'degree': __degree, 'label': __label, 'edge': __edge})
      
        return NETWORK





    NETWORK = graph2NETWORK(graph)
    
    try:
        if ALWAYS_USE_NATIVE_IMPLEMENTATION:
            raise(ImportError('ALWAYS_USE_NATIVE_IMPLEMENTATION'))
        from Modules.estimate_py.c import estimate as __estimate
        
    except ImportError:
        from Modules.estimate_py import estimate as __estimate
        
        warnings.warn('Using the native Python inference method. This will significantly increase the runtime.')
    
    return __estimate.estimate(NETWORK, K=K, K0=K0, MCsweeps=MCsweeps, verbose=verbose)
