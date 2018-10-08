#--------- MAKE SURE TO RUN PROGRAM IN CONSOLE USING module swap python/anaconda2-5.0.1 ---------

#What it does
#By Meteo 473 Z-Team: Alice Tamayev, Alon Sidel, Nicholas Norman, Yushan Han
#Last edited 10/2 - 6PM
#Files edited: MS1_teamZ.py

#Milestone Objective:
"""
Mission  Read the 06Z October 17, 2017 NEXRAD netCDF file from my data folder.  Then plot the following radar fields versus azimuth (i.e. direction) and range: Reflectivity, Radial Velocity, Spectrum Width, Correlation, and Phi (a phase angle). The radial velocity data will need to be de-aliased. Once that is done, you can conduct VAD analyses of all five of these fields. The VAD results will indicate which way the birds are going.
"""
"""
1) 
"""
#Stuff we need to still figure out
"""
1) Aliasing
2) Add polar plot axes? (not sure if this is necessary)
3) VAD 
"""


#  ----- Import required modules -----
import os, sys #Import parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #^
from netCDF4 import Dataset #import netCDF
from scipy.stats import norm #Normalizing function
import numpy as np #Numpy
from mpl_toolkits.basemap import Basemap #Mapping
import matplotlib.pyplot as plt #Plotting
import ncMod as nc  # Nexrad manipulation
import nexradMod as rdr  #Nexrad manipulation
import pandas as pd #"Excel for Python" 
import statsmodels.api as sm #Aliasing
from sklearn import linear_model #Aliasing


#  ----- Configure program -----
# October 17, 2017 at around 6Z
inputPath = '/baldr/s0/g3y/Bird473/Data/Oct_17_2017_CCX/output/20171017/'
inputFile =  'cfrad.20171017_060838.848_to_20171017_061702.970_KCCX_Surveillance_SUR.nc'
maxGate = 600
sweep = 0


#  ----- Open netCDF data file -----
inputFilePath = inputPath+inputFile
print inputFilePath
ncfile=Dataset(inputFilePath,'r')


'''
#  ----- Determine contents of a netCDF file -----
#  Print the ncfile object - shows file attributes, dimension object list and variable object lists 
print ncfile
print ' '
print '--- module-based ncdump ---'
print ' '
nc.ncdump(inputFilePath)
'''

#  ----- Import variables from netCDF ----- 
azimuth, elevation, nyquist_velocity, gate_spacing, start_range = rdr.metaData(ncfile)
ref = rdr.varRead(ncfile,'REF') #Reflectivity
corr = rdr.varRead(ncfile,'RHO') #Correlation
rad_vel = rdr.varRead(ncfile,'VEL') #Radial Velocity
phi = rdr.varRead(ncfile,'PHI') #PHI
spec_width = rdr.varRead(ncfile,'SW') #Spectrum Width
r_range = ncfile.variables['range'][:] #Range (distance from center to data)



#  ----- Plot raw netCDF variables ----- 

# ----- Reshaped data into time-range array -----
beamWidthH = ncfile.variables['radar_beam_width_h'][:]
beamWidthV = ncfile.variables['radar_beam_width_v'][:]
beamSigmaV = beamWidthV/(2.0*1.176)  # convert angle from half power point to half power point into standard deviation of Gaussian

# Start and stop indices for all the sweeps
startIndices = ncfile.variables['sweep_start_ray_index'][:]
endIndices = ncfile.variables['sweep_end_ray_index'][:]
s = startIndices[sweep]
e = endIndices[sweep]
ray_n_gates = ncfile.variables['ray_n_gates']
nGates = ray_n_gates[s] # Pick out the number of gates used for this sweep.
nAzimuths = e-s
azimuth_r = azimuth[s:e]

velArray = np.transpose(np.reshape(rad_vel[0:nAzimuths*nGates],[nAzimuths,nGates]))
refArray = np.transpose(np.reshape(ref[0:nAzimuths*nGates],[nAzimuths,nGates]))
corrArray = np.transpose(np.reshape(corr[0:nAzimuths*nGates],[nAzimuths,nGates]))
phiArray = np.transpose(np.reshape(phi[0:nAzimuths*nGates],[nAzimuths,nGates]))
spec_widthArray = np.transpose(np.reshape(spec_width[0:nAzimuths*nGates],[nAzimuths,nGates]))

all_var = [velArray,refArray,corrArray,phiArray,spec_widthArray]
all_var_nam = ['velArray','refArray','corrArray','phiArray','spec_widthArray']

#Plot variables with unaltered velocity
for i in range(len(all_var)):
	plt.figure()
	plt.imshow(all_var[i][:,:],cmap="Spectral")
	plt.gca().invert_yaxis()
	plt.colorbar(orientation='vertical')
	plt.xlabel('Azimuth Bin')
	plt.ylabel('Range Gate')
	plt.title(all_var_nam[i])
plt.show()

#Compute beam height
meanElevation = np.mean(elevation[s:e])
Re = 6371 # km
IR = 1.21 # refractive index
gateSpacing = np.mean(gate_spacing[s:e])/1000.0
startRange = np.mean(start_range[s:e])/1000.0
rng = ncfile.variables['range'][:]
R = rng[0:maxGate]/1000.0 # Convert to km and truncate to max range that we plan to use
h = (R*np.sin(np.deg2rad(meanElevation))) + np.power(R,2)/(2*IR*Re)

#Dealiasing (given generously by group 5)
VelRing = velArray[0,:]
azsin = np.sin(np.radians(azimuth_r)) #Convert data to trig
azcos = np.cos(np.radians(azimuth_r))

x = np.vstack((azsin,azcos)).T #Linear regression 
b,c = np.linalg.lstsq(x,VelRing)[0] 
VelFit = b*azsin+c*azcos

k = 0 #Loop to fix aliasing by finding multiples of 2*nyquist
for junk in VelRing: 
	if VelRing[k]-VelFit[k] < -nyquist_velocity[0]:  # if data is less than nyquist it is sent to be fixed 
		nshift = np.floor((VelRing[k]-VelFit[k])/(2*nyquist_velocity[0]))  
		VelRing[k] = VelRing[k] - nshift*2*nyquist_velocity[0]

	if VelRing[k]-VelFit[k] > nyquist_velocity[0]:  # if data is more than nyquist it is sent to be fixed 
		nshift = np.ceil((VelRing[k]-VelFit[k])/(2*nyquist_velocity[0]))  
		VelRing[k] = VelRing[k] - nshift*2*nyquist_velocity[0]
	k = k+1
	velArray[0,:] = VelRing 

#Plot aliased vs dealiased velocity
plt.figure()
plt.subplot(1,2,1)
plt.imshow(all_var[0],cmap="Spectral")
plt.gca().invert_yaxis()
plt.xlabel('Azimuth Bin')
plt.ylabel('Range Gate')
plt.title('Aliased (Top) versus Dealiased (Bottom) Velocity)',loc='left')
plt.subplot(1,2,2)
plt.imshow(velArray,cmap="Spectral")
plt.gca().invert_yaxis()
plt.colorbar(orientation='vertical')
plt.show()

all_var[0] = velArray

#VAD Analysis
azimuthRad = np.radians(azimuth_r)
sinAz = np.sin(azimuthRad)
cosAz = np.cos(azimuthRad)
ones = np.ones(np.shape(cosAz))
X = np.transpose([ones,sinAz,cosAz])
x_VAD = np.zeros([maxGate])
y_VAD = np.zeros([maxGate])
mean = np.zeros([maxGate])

for i in range(len(all_var)):
	var_using = all_var[i]
	for rangeGate in range(maxGate):
		mean_az,sinCoef,cosCoef = np.linalg.lstsq(X,var_using[rangeGate])[0]
        	x_VAD[rangeGate] = sinCoef
        	y_VAD[rangeGate] = cosCoef
        	mean[rangeGate] = mean_az
        	
        #Plot VAD Analysis
	plt.figure()
	plt.title('VAD Analysis (Cartesian)')
	plt.plot(x_VAD,h,color='red')
	plt.plot(y_VAD,h,color='blue')
	plt.plot(mean,h,color='black')
	plt.ylabel('Altitude (km)')
	plt.xlabel(all_var_nam[i])
	plt.legend((x_VAD,y_VAD,mean),('x','y','mean'),loc='center right',framealpha=0.4) #WHY ISN'T THIS WORKING
	
	#Cartesian to polar (VAD)
	m = np.sqrt(x_VAD**2.0+y_VAD**2.0)
	d = np.mod(np.rad2deg(np.arctan2(-x_VAD,-y_VAD)),360) # minus sin to get the "from" direction

	#Plot Polar VAD Analysis
	plt.figure()
	plt.title('VAD Analysis (Polar)')
	plt.subplot(1,2,1)
	plt.plot(m,h,color='red')
	plt.ylabel('Altitude (km)')
	plt.xlabel(all_var_nam[i]+' Polar Magnitude')
	plt.subplot(1,2,2)
	plt.plot(d,h,color='blue')
	plt.xlabel(all_var_nam[i]+' polar Direction')
	plt.show()

#Cartesian to polar; R is r_range, Theta is azimuth_r variable (azimuth[s:e])
x_polar = np.matmul(r_range[:,None],np.cos(np.deg2rad(azimuth_r))[None,:])
y_polar = np.matmul(r_range[:,None],np.sin(np.deg2rad(azimuth_r))[None,:])

#Plot polar data
for i in range(len(all_var)):
	plt.figure()
	plt.pcolormesh(x_polar,y_polar,all_var[i],cmap="Spectral")
	plt.colorbar(orientation='vertical')
	plt.title(all_var_nam[i]+' Polar')
plt.show()


