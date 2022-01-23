# Similarity-Graph-Clustering (SGC) assembly detection

Python code implementing the Similarity-Graph-Clustering (SGC) approach to detect neural assemblies in calcium imaging data introduced in

L. Avitan et al. "Spontaneous Activity in the Zebrafish Tectum Reorganizes over Development and Is Influenced by Visual Experience". *Curr. Biol.* **27** (2017). DOI: [10.1016/j.cub.2017.06.056](https://doi.org/10.1016/j.cub.2017.06.056).

For details and a discussion about this approach see

J. Mölter, L. Avitan, G. J. Goodhill. "Detecting neural assemblies in calcium imaging data". *BMC Biol.* **16** (2018). DOI: [10.1186/s12915-018-0606-4](https://doi.org/10.1186/s12915-018-0606-4).

## Overview

This code implements the Similarity-Graph-Clustering (SGC) approach to detect neural assemblies in calcium imaging data in Python as opposed to MATLAB and by that make it available to a wider audience. Importantly, this implementation makes use of the original code by [Mark E. Newman](http://www-personal.umich.edu/~mejn/) to perform the statistical inference of the number of communities present in given graph written in C in the form of a compiled Python extension, which has the decisive advantage that the runtime is significantly reduced.

## Code

This code attempts to mimick the original MATLAB implementation as close as possible. Moreover, for the most part, it operates on the same input files and the output files are compatible to the ones created with MATLAB.

Having said that, full compatibility could not be established in case of the `*_SGC-ASSEMBLIES.mat` files that, coming from the original implementation, hold within their structures a MATLAB graph object. It is not possible to read these in Python using the SciPy libraries. When creating these files in this implementation, instead we save (only) the graph's adjacency matrix in the same field.

To update old files generate with the MATLAB implementation, we suggest using the following function in MATLAB to replace the graph object within the files with the corresponding adjacency matrix.

```matlab
function MATLAB_to_Python(SGC_ASSEMBLIES_file)
%

narginchk(1,1);

SGC_ASSEMBLIES_mat = load(SGC_ASSEMBLIES_file);

SGC_ASSEMBLIES_mat.assembly_pattern_detection.patternSimilarityAnalysis.graph = full(adjacency(SGC_ASSEMBLIES_mat.assembly_pattern_detection.patternSimilarityAnalysis.graph));

save(SGC_ASSEMBLIES_file, '-v7', '-struct', 'SGC_ASSEMBLIES_mat');
```

```matlab
>> MATLAB_to_Python('/path/to/*_SGC-ASSEMBLIES.mat')
```

One thing that is worth remembering when it comes to differences between MATLAB and Python is that Python uses 0-based indexing whereas MATLAB uses 1-based indexing. Hence, in order to maintain maximum compatibility, between the files generated with the different implementation we assume and adopt 1-based indexing in the input and output files, which e.g. affects the way the assemblies are represented.

Apart from these subtleties, everything else should work the same as in the original implementation and we refer to the corresponding repository [GoodhillLab/neural-assembly-detection](https://github.com/GoodhillLab/neural-assembly-detection) for a comprehensive description of the input and output files.

The implementation has been tested in a Python 3.9 environment together with the NumPy, SciPy, Scikit-Learn, and NetworkX packages. 

## Usage

The functions to perform the preprocessing of the calcium fluorescence signals as well as the assembly detection can be accessed via the common interface provided by the `SGC.py` code file.

### Preprocessing

```bash
python SGC.py preprocessing /path/to/*_CALCIUM-FLUORESCENCE.mat
```

This step produces an `*_ACTIVITY-RASTER.mat` file that holds the binary activity patterns and which is used to perform the assembly detection; see [GoodhillLab/neural-assembly-detection](https://github.com/GoodhillLab/neural-assembly-detection) for details.

### Assembly detection

```bash
python SGC.py detection /path/to/*_ACTIVITY-RASTER.mat
```

This step produces an `*_SGC-ASSEMBLIES.mat` file that holds the results of the assembly detection; see [GoodhillLab/neural-assembly-detection](https://github.com/GoodhillLab/neural-assembly-detection) for details.

Note: In order to efficiently run the assembly detection, the procedure to perform the statistical inference for the number of communities present in a graph, has to be compiled first. On Linux and macOS this should be straightforward, but requires Python C-development libraries as well as the GNU Scientific Library (GSL) to be present.

```bash
(cd Modules/estimate_py/c/; make)
```

## References

L. Avitan et al. "Spontaneous Activity in the Zebrafish Tectum Reorganizes over Development and Is Influenced by Visual Experience". *Curr. Biol.* **27** (2017). DOI: [10.1016/j.cub.2017.06.056](https://doi.org/10.1016/j.cub.2017.06.056).

J. Mölter, L. Avitan, G. J. Goodhill. "Detecting neural assemblies in calcium imaging data". *BMC Biol.* **16** (2018). DOI: [10.1186/s12915-018-0606-4](https://doi.org/10.1186/s12915-018-0606-4).

M. E. J. Newman, G. Reinert. "Estimating the Number of Communities in a Network". *Phys. Rev. Lett.* **117** (2016). DOI: [10.1103/PhysRevLett.117.078301](https://doi.org/10.1103/PhysRevLett.117.078301).
