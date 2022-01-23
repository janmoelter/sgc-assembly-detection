#include <Python.h>

#if PY_MAJOR_VERSION < 3
#error "Requires Python 3"
#include "stopcompilation"
#endif

#include <stdbool.h>

// =============================================================================
// estimate.c ==================================================================
// Note: '___' has been added as a prefix to all functions
// =============================================================================

/* Program to calculate the number of communities in a network using the
 * method of Newman and Reinert, which calculates a posterior probability
 * by Monte Carlo simulation of the integrated likelihood of a
 * degree-corrected stochastic block model
 *
 * Written by Mark Newman  6 APR 2016
 */

/* Inclusions */

//#include <stdio.h>
//#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_gamma.h>

#include <network.h>

/* Program control */

//#define VERBOSE
int VERBOSE = 1;

int RNG_SEED = 0;

/* Constants */

//#define K 40		   // Maximum number of groups
int K = 40;
//#define K0 40		   // Initial number of groups
int K0 = 40;
//#define MCSWEEPS 10000 // Number of Monte Carlo sweeps
int MCSWEEPS = 10000;
//#define SAMPLE 1	   // Interval at which to print out results, in sweeps
int SAMPLE = 10;

int *k__;
double *E__;

/* Globals */

NETWORK G; // Struct storing the network
int twom;  // Twice the number of edges
double p;  // Average edge probability

int k;	 // Current value of k
int *g;	 // Group assignments
int *n;	 // Group sizes
int **m; // Edge counts

double *lnfact; // Look-up table of log-factorials
double E;		// Log probability

gsl_rng *rng; // Random number generator

// Make a lookup table of log-factorial values

void ___maketable()
{
	int t;
	int length;

	length = twom + G.nvertices + 1;
	lnfact = malloc(length * sizeof(double));
	for (t = 0; t < length; t++)
		lnfact[t] = gsl_sf_lnfact(t);
}

// Log-probability function

double ___logp(int *n, int **m)
{
	int r, s;
	int kappa;
	double res = 0.0;

	for (r = 0; r < k; r++)
	{
		res += lnfact[n[r]];
		if (n[r] > 0)
		{
			for (s = kappa = 0; s < k; s++)
				kappa += m[r][s];
			res += kappa * log(n[r]) + lnfact[n[r] - 1] - lnfact[kappa + n[r] - 1];
			res += lnfact[m[r][r] / 2] - (m[r][r] / 2 + 1) * log(0.5 * p * n[r] * n[r] + 1);
			for (s = r + 1; s < k; s++)
			{
				res += lnfact[m[r][s]] - (m[r][s] + 1) * log(p * n[r] * n[s] + 1);
			}
		}
	}

	return res;
}

// Initial group assignment

void ___initgroups()
{
	int i, u, v;
	int r;

	// Make the initial group assignments at random

	g = malloc(G.nvertices * sizeof(int));
	for (u = 0; u < G.nvertices; u++)
		g[u] = gsl_rng_uniform_int(rng, K0);

	// Calculate the values of the n's

	n = calloc(K, sizeof(int));
	for (u = 0; u < G.nvertices; u++)
		n[g[u]]++;

	// Calcalate the values of the m's

	m = malloc(K * sizeof(int *));
	for (r = 0; r < K; r++)
		m[r] = calloc(K, sizeof(int));
	for (u = 0; u < G.nvertices; u++)
	{
		for (i = 0; i < G.vertex[u].degree; i++)
		{
			v = G.vertex[u].edge[i].target;
			m[g[u]][g[v]]++;
		}
	}

	// Initialize k and the log-probability

	k = K0;
	E = ___logp(n, m);
}

// Function to update value of k

void ___changek()
{
	int r, s, u;
	//int kp;
	int empty;
	//int map[K];
	//int sum;

	// With probability 0.5, decrease k, otherwise increase it

	if (gsl_rng_uniform(rng) < 0.5)
	{

		// Count the number of empty groups

		for (r = 0, empty = 0; r < k; r++)
			if (n[r] == 0)
				empty++;

		// If there are any empty groups, remove one of them, or otherwise do
		// nothing

		if (empty > 0)
		{

			// If there is more than one empty group, choose at random which one
			// to remove

			do
			{
				r = gsl_rng_uniform_int(rng, k);
			} while (n[r] > 0);

			// Decrease k by 1

			k = k - 1;

			// Update the group labels

			for (u = 0; u < G.nvertices; u++)
			{
				if (g[u] == k)
					g[u] = r;
			}

			// Update n_r

			n[r] = n[k];

			// Update m_rs

			for (s = 0; s < k; s++)
			{
				if (r == s)
				{
					m[r][r] = m[k][k];
				}
				else
				{
					m[r][s] = m[k][s];
					m[s][r] = m[s][k];
				}
			}
		}
	}
	else
	{

		// With probability k/(n+k) increase k by 1, adding an empty group

		if ((G.nvertices + k) * gsl_rng_uniform(rng) < k)
		{
			if (k < K)
			{
				n[k] = 0;
				for (r = 0; r <= k; r++)
					m[k][r] = m[r][k] = 0;
				k = k + 1;
			}
		}
	}
}

// Function to update n and m for a proposed move

void ___nmupdate(int r, int s, int d[])
{
	int t;

	n[r]--;
	n[s]++;
	for (t = 0; t < k; t++)
	{
		m[r][t] -= d[t];
		m[t][r] -= d[t];
		m[s][t] += d[t];
		m[t][s] += d[t];
	}
}

// Function that does one MCMC sweep (i.e., n individual moves) using the
// heatbath algorithm

double ___sweep()
{
	int i, j, u, v;
	int r, s;
	//int temp;
	int accept = 0;
	int d[K];
	double x, Z, sum;
	double newE[K];
	double boltzmann[K];

	for (i = 0; i < G.nvertices; i++)
	{

		// Optionally, perform a k-changing move

		if ((G.nvertices + 1) * gsl_rng_uniform(rng) < 1.0)
			___changek();

		// Choose a random node

		u = gsl_rng_uniform_int(rng, G.nvertices);
		r = g[u];

		// Find the number of edges this node has to each group

		for (s = 0; s < k; s++)
			d[s] = 0;
		for (j = 0; j < G.vertex[u].degree; j++)
		{
			v = G.vertex[u].edge[j].target;
			d[g[v]]++;
		}

		// Calculate the probabilities of moving it to each group in turn

		Z = 0.0;
		for (s = 0; s < k; s++)
		{
			if (s == r)
			{
				newE[s] = E;
			}
			else
			{
				___nmupdate(r, s, d);
				newE[s] = ___logp(n, m);
				___nmupdate(s, r, d);
			}
			boltzmann[s] = exp(newE[s] - E);
			Z += boltzmann[s];
		}

		// Choose which move to make based on these probabilities

		x = Z * gsl_rng_uniform(rng);
		for (s = 0, sum = 0.0; s < k; s++)
		{
			sum += boltzmann[s];
			if (sum > x)
				break;
		}

		// Make the move

		if (s != r)
		{
			g[u] = s;
			___nmupdate(r, s, d);
			E = newE[s];
			accept++;
		}
	}

	return (double)accept / G.nvertices;
}

int ___main()
{
	//int u, r, s;
	int u, s;


	k__ = malloc(MCSWEEPS * sizeof(int));
	E__ = malloc(MCSWEEPS * sizeof(double));

	// Initialize the random number generator from the system clock

	rng = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(rng, RNG_SEED);

	// Read the network from stdin

#ifdef VERBOSE
	fprintf(stderr, "Reading network...\n");
#endif
//	read_network(&G, stdin);
	for (u = twom = 0; u < G.nvertices; u++) {
		twom += G.vertex[u].degree;
	}
	p = (double)twom / (G.nvertices * G.nvertices);
//#ifdef VERBOSE
//	fprintf(stderr, "Read network with %i nodes and %i edges\n",
//			G.nvertices, twom / 2);
//#endif

	// Make the lookup table

	___maketable();

	// Initialize the group assignment

	___initgroups();

	// Perform the Monte Carlo

	for (s = 0; s < MCSWEEPS; s++)
	{
		___sweep();
		k__[s] = k;
		E__[s] = E;

		if (s % SAMPLE == 0)
		{
			//printf("%i %i %g\n", s, k, E);
#ifdef VERBOSE
			fprintf(stderr, "Sweep %i...\r", s);
#endif
		}
	}
#ifdef VERBOSE
	fprintf(stderr, "\n");
#endif

	return 0;
}

// =============================================================================
// END: estimate.c =============================================================
// =============================================================================

//int test__estimate()
//{
//	k__ = malloc(MCSWEEPS * sizeof(int));
//	E__ = malloc(MCSWEEPS * sizeof(double));
//
//	rng = gsl_rng_alloc(gsl_rng_mt19937);
//	gsl_rng_set(rng, RNG_SEED);
//
//	int s;
//	for (s = 0; s < MCSWEEPS; s++)
//	{
//		k__[s] = gsl_rng_uniform_int(rng, K);
//		E__[s] = gsl_rng_uniform(rng);
//	}
//
//	return 0;
//}

// =============================================================================
// Python Control Structures ===================================================
// =============================================================================

static PyObject* estimate_func(PyObject *self, PyObject *args, PyObject *kwargs)
{
	static char* argnames[] = {"graph", "_", "K", "K0", "MCsweeps", "seed", "verbose", NULL};

	PyObject* arg_graphGMLdict;
	int arg__ = 0;
    int arg_K = 40;
	int arg_K0 = 40;
	int arg_MCSWEEPS = 10000;
	int arg_seed = time(NULL);
	bool arg_verbose = true;

    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|iiiiip", argnames, &PyDict_Type, &arg_graphGMLdict, &arg__, &arg_K, &arg_K0, &arg_MCSWEEPS, &arg_seed, &arg_verbose))
	{
        return NULL;
	}

    //PyObject_Print(kwargs, std_out, 0);
    //fprintf(std_out, "\n");	

	VERBOSE = arg_verbose;
	K = arg_K;
	K0 = (arg_K < arg_K0) ? arg_K : arg_K0;
	MCSWEEPS = arg_MCSWEEPS;
	RNG_SEED = arg_seed;


	PyObject *py_G_nvertices = PyDict_GetItemWithError(arg_graphGMLdict, PyUnicode_FromString("nvertices"));

	G.nvertices = (int)PyLong_AsLong(py_G_nvertices);
	G.vertex = calloc(G.nvertices, sizeof(VERTEX));

	PyObject *py_G_vertex = PyDict_GetItemWithError(arg_graphGMLdict, PyUnicode_FromString("vertex"));

	int i, j;

	for(i = 0; i < G.nvertices; i++)
	{
		PyObject *py_G_vertex_i = PyList_GetItem(py_G_vertex, i);

		PyObject *py_G_vertex_i_id = PyDict_GetItemWithError(py_G_vertex_i, PyUnicode_FromString("id"));
		PyObject *py_G_vertex_i_degree = PyDict_GetItemWithError(py_G_vertex_i, PyUnicode_FromString("degree"));
		PyObject *py_G_vertex_i_label = PyDict_GetItemWithError(py_G_vertex_i, PyUnicode_FromString("label"));

		G.vertex[i].id = (int)PyLong_AsLong(py_G_vertex_i_id);
		G.vertex[i].degree = (int)PyLong_AsLong(py_G_vertex_i_degree);
		G.vertex[i].label = NULL;

		G.vertex[i].edge = calloc(G.vertex[i].degree, sizeof(EDGE));
		
		PyObject *py_G_vertex_i_edge = PyDict_GetItemWithError(py_G_vertex_i, PyUnicode_FromString("edge"));

		for(j = 0; j < G.vertex[i].degree; j++)
		{
			PyObject *py_G_vertex_i_edge_j = PyList_GetItem(py_G_vertex_i_edge, j);

			PyObject *py_G_vertex_i_edge_j_target = PyDict_GetItemWithError(py_G_vertex_i_edge_j, PyUnicode_FromString("target"));
			PyObject *py_G_vertex_i_edge_j_weight = PyDict_GetItemWithError(py_G_vertex_i_edge_j, PyUnicode_FromString("weight"));

			G.vertex[i].edge[j].target = (int)PyLong_AsLong(py_G_vertex_i_edge_j_target);
			G.vertex[i].edge[j].weight = (int)PyLong_AsLong(py_G_vertex_i_edge_j_weight);
		}
	}


	___main();


	PyObject *py_k = PyList_New(MCSWEEPS);
	PyObject *py_E = PyList_New(MCSWEEPS);

	for (int s = 0; s < MCSWEEPS; s++)
	{
		PyList_SetItem(py_k, s, PyLong_FromLong(k__[s]));
		PyList_SetItem(py_E, s, PyFloat_FromDouble(E__[s]));
	}

	PyObject *py_out = PyTuple_New(2);
	PyTuple_SetItem(py_out, 0, py_k);
	PyTuple_SetItem(py_out, 1, py_E);

	return py_out;
}

static PyMethodDef estimateMethods[] = {
	{"estimate", estimate_func, METH_VARARGS | METH_KEYWORDS, "<estimate.__doc__>"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef estimate_ =
{
	PyModuleDef_HEAD_INIT, "estimate" ,"" , -1, estimateMethods
};

PyMODINIT_FUNC PyInit_estimate(void)
{
	return PyModule_Create(&estimate_);
}

// =============================================================================
// END: Python Control Structures ==============================================
// =============================================================================
