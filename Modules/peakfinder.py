
import math
import numpy as np
import warnings


def sign(x):
	if not isinstance(x, np.ndarray):
		if not x == 0:
			return int(x / abs(x))
		else:
			return 0
	else:
		return np.array(list(map(sign, x)))
	
def isnumeric(x):
	if not isinstance(x, np.ndarray):
		return isinstance(x, (int,float,complex))
	else:
		return np.array(list(map(isnumeric, x)))


def peakfinder(x0, sel=None, thresh=None, extrema=1, includeEndpoints=True, interpolate=False, plot=False):
	"""
	PEAKFINDER Noise tolerant fast peak finding algorithm
	   INPUTS:
	       x0 - A real vector from the maxima will be found (required)
	       sel - The amount above surrounding data for a peak to be,
	           identified (default = (max(x0)-min(x0))/4). Larger values mean
	           the algorithm is more selective in finding peaks.
	       thresh - A threshold value which peaks must be larger than to be
	           maxima or smaller than to be minima.
	       extrema - 1 if maxima are desired, -1 if minima are desired
	           (default = maxima, 1)
	       includeEndpoints - If true the endpoints will be included as
	           possible extrema otherwise they will not be included
	           (default = True)
	       interpolate - If true quadratic interpolation will be performed
	           around each extrema to estimate the magnitude and the
	           position of the peak in terms of fractional indicies. Note that
	           unlike the rest of this function interpolation assumes the
	           input is equally spaced. To recover the x_values of the input
	           rather than the fractional indicies you can do:
	           peakX = x0 + (peakLoc - 1) * dx
	           where x0 is the first x value and dx is the spacing of the
	           vector. Output peakMag to recover interpolated magnitudes.
	           See example 2 for more information.
	           (default = False)
	       plot - If true the identified maxima will also be plotted along with
	           the input data.
	
	   OUTPUTS:
	       peakLoc - The indicies of the identified peaks in x0
	       peakMag - The magnitude of the identified peaks
	
	   peakLoc, _ = peakfinder(x0) returns the indicies of local maxima that
	       are at least 1/4 the range of the data above surrounding data.
	
	   peakLoc, _ = peakfinder(x0,sel) returns the indicies of local maxima
	       that are at least sel above surrounding data.
	
	   peakLoc, _ = peakfinder(x0,sel,thresh) returns the indicies of local
	       maxima that are at least sel above surrounding data and larger
	       (smaller) than thresh if you are finding maxima (minima).
	
	   peakLoc, _ = peakfinder(x0,sel,thresh,extrema) returns the maxima of the
	       data if extrema > 0 and the minima of the data if extrema < 0
	
	   peakLoc, _ = peakfinder(x0,sel,thresh,extrema, includeEndpoints)
	       returns the endpoints as possible extrema if includeEndpoints is
	       considered true in a boolean sense
	
	   peakLoc, peakMag = peakfinder(x0,sel,thresh,extrema,interpolate)
	       returns the results of results of quadratic interpolate around each
	       extrema if interpolate is considered to be true in a boolean sense
	
	   peakLoc, peakMag = peakfinder(x0,...) returns the indicies of the
	       local maxima as well as the magnitudes of those maxima
	
	
	   Note: If repeated values are found the first is identified as the peak
	
	Example 1:
	t = np.linspace(0,10,100000+1);
	x = 12*np.sin(10*2*np.pi*t)-3*np.sin(.1*2*np.pi*t)+np.random.normal(size=len(t));
	x[(1250-1):1255] = max(x);
	peakfinder(x, plot=False)
	
	Example 2:
	ds = 100;  # Downsample factor
	dt = .001; # Time step
	ds_dt = ds*dt; # Time delta after downsampling
	t0 = 1;
	t = np.arange(t0,5+dt + t0,dt);
	x = 0.2-np.sin(0.01*2*np.pi*t)+3*np.cos(7/13*2*np.pi*t+.1)-2*np.cos((1+np.pi/10)*2*np.pi*t+0.2)-0.2*t;
	x[-1] = x.min();
	x_ds = x[::ds]; # Downsample to test interpolation
	minLoc, minMag = peakfinder(x_ds, .8, 0, -1, False, True);
	minT = t0 + minLoc * ds_dt;
	plt.plot(t, x, color='black', label='Actual Data', zorder=1)
	plt.scatter(t[::ds], x_ds, color='black', marker='o', label='Input Data', zorder=2)
	plt.scatter(minT, minMag, color='red', marker='v', label='Estimated Peaks', zorder=3)
	plt.legend()
	
	Original MATLAB implementation:
	 Copyright Nathanael C. Yoder 2015 (nyoder@gmail.com)
	"""
	
	# s = x0.shape
	# if len(s) > 1:
	# 	flipData = s[0] < s[1]
	# else:
	# 	flipData = False
	len0 = len(x0);
	if not x0.shape == (len0,):
		raise ValueError('PEAKFINDER:Input : The input data must be a vector')
	elif len0 == 0:
		return [np.array([]), np.array([])];
	
	if not(np.all(np.isreal(x0))):
		warnings.warn('PEAKFINDER:NotReal : Absolute value of data will be used')
		x0 = np.abs(x0);
	
	if sel is None:
		sel = (x0.max() - x0.min())/4;
	elif not(isnumeric(sel)) or not(np.isreal(sel)):
		sel = (x0.max() - x0.min())/4;
		warning('PEAKFINDER:InvalidSel : The selectivity must be a real scalar.  A selectivity of {:.3f} will be used'.format(sel))
	elif isinstance(sel, (list, np.ndarray)) and len(sel) > 1:
		warning('PEAKFINDER:InvalidSel : The selectivity must be a scalar.  The first selectivity value in the vector will be used.')
		sel = sel[0];
		
	if thresh is None:
		pass
	elif not(isnumeric(thresh)) or not(np.isreal(thresh)):
		thresh = None;
		warning('PEAKFINDER:InvalidThreshold : The threshold must be a real scalar. No threshold will be used.')
	elif isinstance(thresh, (list, np.ndarray)) and len(thresh) > 1:
		warning('PEAKFINDER:InvalidThreshold : The threshold must be a scalar.  The first threshold value in the vector will be used.')
		thresh = thresh[0];
		
	if extrema is None:
		extrema = 1;
	else:
		if extrema == 0:
			raise ValueError('PEAKFINDER:ZeroMaxima : Either 1 (for maxima) or -1 (for minima) must be input for extrema')
		else:
			extrema = sign(extrema);
			
	if includeEndpoints is None:
		includeEndpoints = True;
		
	if interpolate is None:
		interpolate = False;
		
		
		
	x0 = extrema*x0; # Make it so we are finding maxima regardless
	if thresh is not None:
		thresh = thresh*extrema; # Adjust threshold according to extrema.
	dx0 = np.diff(x0); # Find derivative
	dx0[dx0 == 0] = -np.finfo(np.float32).eps; # This is so we find the first of repeated values
	ind = np.where(dx0[:-1]*dx0[1:] < 0)[0]+1; # Find where the derivative changes sign
	
	# Include endpoints in potential peaks and valleys as desired
	if includeEndpoints:
		x = np.concatenate([np.array([x0[0]]),x0[ind],np.array([x0[-1]])]);
		ind = np.concatenate([np.array([0]),ind,np.array([len0])]);
		minMag = x.min();
		leftMin = minMag;
	else:
		x = x0[ind];
		minMag = x.min();
		leftMin = min([x[0], x0[0]]);
	
	# x only has the peaks, valleys, and possibly endpoints
	len_ = len(x);
	
	if len_ > 2: # Function with peaks and valleys
		# Set initial parameters for loop
		tempMag = minMag;
		foundPeak = False;
		
		if includeEndpoints:
			# Deal with first point a little differently since tacked it on
			# Calculate the sign of the derivative since we tacked the first
			#  point on it does not neccessarily alternate like the rest.
			signDx = sign(np.diff(x[0:3]));
			if signDx[0] <= 0: # The first point is larger or equal to the second
				if signDx[0] == signDx[1]: # Want alternating signs
					x = np.delete(x, 1);
					ind = np.delete(ind, 1);
					len_ = len_-1;
			else: # First point is smaller than the second
				if signDx[0] == signDx[1]: # Want alternating signs
					x = np.delete(x, 0);
					ind = np.delete(ind, 0);
					len_ = len_-1;
		
		# Skip the first point if it is smaller so we always start on a
		#   maxima
		if x[0] >= x[1]:
			ii = -1;
		else:
			ii = 0;
		
		# Preallocate max number of maxima
		maxPeaks = math.ceil(len_/2);
		peakLoc = np.zeros(maxPeaks, dtype=int);
		peakMag = np.zeros(maxPeaks, dtype=float);
		cInd = 0;
		# Loop through extrema which should be peaks and then valleys
		while ii < len_-1:
			ii = ii+1; # This is a peak
			# Reset peak finding if we had a peak and the next peak is bigger
			#   than the last or the left min was small enough to reset.
			if foundPeak:
				tempMag = minMag;
				foundPeak = False;
			
			# Found new peak that was lager than temp mag and selectivity larger
			#   than the minimum to its left.
			if x[ii] > tempMag and x[ii] > leftMin + sel:
				tempLoc = ii;
				tempMag = x[ii];
			
			# Make sure we don't iterate past the length of our vector
			if ii == len_-1:
				break; # We assign the last point differently out of the loop
			
			ii = ii+1; # Move onto the valley
			# Come down at least sel from peak
			if not(foundPeak) and tempMag > sel + x[ii]:
				foundPeak = True; # We have found a peak
				leftMin = x[ii];
				peakLoc[cInd] = tempLoc; # Add peak to index
				peakMag[cInd] = tempMag;
				cInd = cInd+1;
			elif x[ii] < leftMin: # New left minima
				leftMin = x[ii];
		
		# Check end point
		if includeEndpoints:
			if x[-1] > tempMag and x[-1] > leftMin + sel:
				peakLoc[cInd] = len_;
				peakMag[cInd] = x[-1];
				cInd = cInd + 1;
			elif not(foundPeak) and tempMag > minMag: # Check if we still need to add the last point
				peakLoc[cInd] = tempLoc;
				peakMag[cInd] = tempMag;
				cInd = cInd + 1;
		elif not(foundPeak):
			if x[-1] > tempMag and x[-1] > leftMin + sel:
				peakLoc[cInd] = len_;
				peakMag[cInd] = x[-1];
				cInd = cInd + 1;
			elif tempMag > min([x0[-1], x[-1]]) + sel:
				peakLoc[cInd] = tempLoc;
				peakMag[cInd] = tempMag;
				cInd = cInd + 1;
		
		# Create output
		if cInd > 0:
			peakInds = ind[peakLoc[:cInd]];
			peakMags = peakMag[:cInd];
		else:
			peakInds = np.array([]);
			peakMags = np.array([]);
	else: # This is a monotone function where an endpoint is the only peak
		peakMags, xInd = np.max(x), np.argmax(x);
		if includeEndpoints and peakMags > minMag + sel:
			peakInds = ind[xInd];
		else:
			peakMags = np.array([]);
			peakInds = np.array([]);
	
	# Apply threshold value.  Since always finding maxima it will always be
	#   larger than the thresh.
	if thresh is not None:
		m = peakMags>thresh;
		peakInds = peakInds[m];
		peakMags = peakMags[m];
	
	if interpolate and len(peakMags) > 0:
		middleMask = np.logical_and(peakInds > 1, peakInds < len0);
		noEnds = peakInds[middleMask];
		
		magDiff = x0[noEnds + 1] - x0[noEnds - 1];
		magSum = x0[noEnds - 1] + x0[noEnds + 1]  - 2 * x0[noEnds];
		magRatio = magDiff / magSum;
		
		peakInds[middleMask] = peakInds[middleMask] - magRatio/2;
		peakMags[middleMask] = peakMags[middleMask] - magRatio * magDiff/8;
	
	# Rotate data if needed
	# if flipData:
	# 	peakMags = peakMags.T;
	# 	peakInds = peakInds.T;
	
	# Change sign of data if was finding minima
	if extrema < 0:
		peakMags = -peakMags;
		x0 = -x0;
	
	# Plot if no output desired
	if plot:
		if len(peakInds) == 0:
			print('No significant peaks found');
		else:
			import matplotlib.pyplot as plt

			plt.plot(np.arange(len0), x0, color='black', zorder=1);
			plt.scatter(peakInds, peakMags, color='red', edgecolors='red', facecolors='none', zorder=2);
	else:
		pass
	
	return peakInds, peakMags;
