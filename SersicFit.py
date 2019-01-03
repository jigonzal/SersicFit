# import matplotlib as mpl
# mpl.use('Agg')
import numpy
import scipy.ndimage
from astropy.stats import sigma_clip
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
from astropy.coordinates import ICRS
from astropy import units as u
# from astropy import wcs
from astropy.io import fits
import os,sys
from scipy import stats
from collections import Counter
import seaborn as sns
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.modeling.models import Gaussian2D
from sklearn.cluster import DBSCAN
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from matplotlib.patches import Rectangle
from astropy.io import ascii
from astropy.cosmology import Planck13 as cosmology
import itertools
import numpy as np
import aplpy
import os.path
import emcee
sns.set_style("white", {'legend.frameon': True})
sns.set_style("ticks", {'legend.frameon': True})
sns.set_context("talk")
sns.set_palette('Dark2',desat=1)
cc = sns.color_palette()
from scipy.special import gammaincinv
from astropy.modeling import models, fitting
from astropy.convolution import convolve,convolve_fft,Kernel2D
import time
import corner
import yaml
from astropy.nddata import Cutout2D
import argparse


def lnprior(theta):
	if theta[0]<ConfigFile['AmplitudeBounds'][0] or theta[0]>ConfigFile['AmplitudeBounds'][3]:
		return -np.inf
	# if theta[1]<pix_size or theta[1]<ConfigFile['ReffBounds'][0] or theta[1]>ConfigFile['ReffBounds'][3]:
	if theta[1]<ConfigFile['ReffBounds'][0] or theta[1]>ConfigFile['ReffBounds'][3]:
		return -np.inf
	if theta[2]<ConfigFile['nBounds'][0] or theta[2]>ConfigFile['nBounds'][3]:
		return -np.inf
	if theta[3]<ConfigFile['x0Bounds'][0] or theta[3]>ConfigFile['x0Bounds'][3]:
		return -np.inf
	if theta[4]<ConfigFile['y0Bounds'][0] or theta[4]>ConfigFile['y0Bounds'][3]:
		return -np.inf
	if theta[5]<ConfigFile['ellipBounds'][0] or theta[5]>ConfigFile['ellipBounds'][3]:
		return -np.inf
	if theta[6]<ConfigFile['thetaBounds'][0] or theta[6]>ConfigFile['thetaBounds'][3]:
		return -np.inf
	return 0.0



def GetKernel(CubePath):
    hdulist =   fits.open(CubePath,memmap=True)
    head = hdulist[0].header
    data = hdulist[0].data[0]

    try:
        BMAJ = hdulist[1].data.field('BMAJ')
        BMIN = hdulist[1].data.field('BMIN')
        BPA = hdulist[1].data.field('BPA')
    except:
        BMAJ = []
        BMIN = []
        BPA = []
        for i in range(len(data)):
            BMAJ.append(head['BMAJ']*3600.0)
            BMIN.append(head['BMIN']*3600.0)
            BPA.append(head['BPA'])
        BMAJ = np.array(BMAJ)
        BMIN = np.array(BMIN)
        BPA = np.array(BPA)

    pix_size = head['CDELT2']*3600.0
    factor = 2*(np.pi*BMAJ*BMIN/(8.0*np.log(2)))/(pix_size**2)
    factor = 1.0/factor
    FractionBeam = 1.0/np.sqrt(2.0)
    FractionBeam = 1.0
    # print 'Fraction Beam',FractionBeam
    KernelList = []

    for i in range(len(BMAJ)):
        SigmaPixel = int((BMAJ[i]*FractionBeam/2.355)/pix_size)+1
        x = np.arange(-(3*SigmaPixel), (3*SigmaPixel))
        y = np.arange(-(3*SigmaPixel), (3*SigmaPixel))        
        x, y = np.meshgrid(x, y)
        arr = models.Gaussian2D(amplitude=1.0,x_mean=0,y_mean=0,
        						x_stddev=(BMAJ[i]*FractionBeam/2.355)/pix_size,
        						y_stddev=(BMIN[i]*FractionBeam/2.355)/pix_size,
        						theta=(BPA[i]*2.0*np.pi/360.0)+np.pi/2)(x,y)
        kernel = Kernel2D(model=models.Gaussian2D(amplitude=1.0,x_mean=0,y_mean=0,
        					x_stddev=(BMAJ[i]*FractionBeam/2.355)/pix_size,
        					y_stddev=(BMIN[i]*FractionBeam/2.355)/pix_size,
        					theta=(BPA[i]*2.0*np.pi/360.0)+np.pi/2),
        					array=arr,width=len(x))
        KernelList.append(kernel)

    return KernelList[0],pix_size,1.0/factor

def lnlike(theta, measured_flux,sigma):
    y_model = Sersic2D(measured_flux, *theta)
    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (measured_flux - y_model) ** 2 / sigma ** 2)

def lnprob(theta, measured_flux,sigma):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, measured_flux,sigma)

def Sersic2D(Array, amplitude, r_eff, n, x_0, y_0, ellip, theta):
	"""Two dimensional Sersic profile function."""

	amplitude = 0.5*amplitude*1e-3  # To mJy and to total flux.
	theta = theta - 90.0
	theta = theta*np.pi/180.0
	r_eff = r_eff/pix_size
	xx = np.arange(len(Array))
	yy = np.arange(len(Array[0]))
	x, y = np.meshgrid(xx, yy)
	bn = gammaincinv(2. * n, 0.5)
	a, b = r_eff, (1 - ellip) * r_eff
	cos_theta, sin_theta = np.cos(theta), np.sin(theta)
	x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
	x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
	z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

	aux = amplitude * np.exp(-bn * (z ** (1 / n) - 1))
	smooth = convolve_fft(aux, Kernel2Use,allow_huge=True)
	smooth = smooth*amplitude/np.max(smooth)
	return smooth

def Sersic2DWithoutConvolving(Array, amplitude, r_eff, n, x_0, y_0, ellip, theta):
	"""Two dimensional Sersic profile function."""

	amplitude = 0.5*amplitude*1e-3  # To mJy and to total flux.
	theta = theta - 90.0
	theta = theta*np.pi/180.0
	r_eff = r_eff/pix_size
	xx = np.arange(len(Array))
	yy = np.arange(len(Array[0]))
	x, y = np.meshgrid(xx, yy)
	bn = gammaincinv(2. * n, 0.5)
	a, b = r_eff, (1 - ellip) * r_eff
	cos_theta, sin_theta = np.cos(theta), np.sin(theta)
	x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
	x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
	z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

	aux = amplitude * np.exp(-bn * (z ** (1 / n) - 1))
	return aux

def prune(samples,lnprob, scaler=5.0, quiet=False):

    minlnprob = lnprob.max()
    dlnprob = numpy.abs(lnprob - minlnprob)
    medlnprob = numpy.median(dlnprob)
    avglnprob = numpy.mean(dlnprob)
    skewlnprob = numpy.abs(avglnprob - medlnprob)
    rmslnprob = numpy.std(dlnprob)
    inliers = (dlnprob < scaler*rmslnprob)
    lnprob2 = lnprob[inliers]
    samples = samples[inliers]

    medlnprob_previous = 0.
    while skewlnprob > 0.1*medlnprob:
        minlnprob = lnprob2.max()
        dlnprob = numpy.abs(lnprob2 - minlnprob)
        rmslnprob = numpy.std(dlnprob)
        inliers = (dlnprob < scaler*rmslnprob)
        PDFdatatmp = lnprob2[inliers]
        if len(PDFdatatmp) == len(lnprob2):
            inliers = (dlnprob < scaler/2.*rmslnprob)
        lnprob2 = lnprob2[inliers]
        samples = samples[inliers]
        dlnprob = numpy.abs(lnprob2 - minlnprob)
        medlnprob = numpy.median(dlnprob)
        avglnprob = numpy.mean(dlnprob)
        skewlnprob = numpy.abs(avglnprob - medlnprob)
        if not quiet:
            print(medlnprob, avglnprob, skewlnprob)
        if medlnprob == medlnprob_previous:
            scaler /= 1.5
        medlnprob_previous = medlnprob
    samples = samples[lnprob2 <= minlnprob]
    lnprob2 = lnprob2[lnprob2 <= minlnprob]
    return samples,lnprob2

def CreateConfigFile():
    cmd = "File: test.fits\nRA: 0.0\nDEC: 0.0\nBoxSize: 2.0 #arcsec\nAmplitudeBounds: [0,0,15.0,15.0] #mJy\nReffBounds: [0.01,0.01,0.5,10.0] #arcsec\nnBounds: [0.5,0.5,5,20.0]\nx0Bounds: [-1,-0.1,0.1,1.0]   #arcsec\ny0Bounds: [-1,-0.1,0.1,1.0]   #arcsec\nellipBounds: [0,0.0,0.9,0.9]\nthetaBounds: [0,0,180,180.0]\nMCMCA: 5.0\nNthreads: 10\nSigmaValue: 0.0116 #mJy\nnwalkers: 100 \nndim: 7\nnsteps: 200"
    output = open('configSersicFit.yaml','w')
    output.write(cmd)
    output.close()
##################################################################################################################
##################################################################################################################
##################################################################################################################

#Parse the input arguments
parser = argparse.ArgumentParser(description="Python script that fit an Sersic profile to an ALMA continuum image")
parser.add_argument('--CreateConfigFile', action='store_true',required=False,help = 'Create template configuration file')


args = parser.parse_args()
#Checking input arguments
print 20*'#','Checking inputs....',20*'#'
if args.CreateConfigFile:
    CreateConfigFile()
    print '*** Creating configuration file ***'



ConfigFile = yaml.safe_load(open('configSersicFit.yaml'))


Bounds = 	[
			'AmplitudeBounds',
			'ReffBounds',
			'nBounds',
			'x0Bounds',
			'y0Bounds',
			'ellipBounds',
			'thetaBounds'
			]

for b in Bounds:
	if ConfigFile[b][1]<ConfigFile[b][0] or ConfigFile[b][1]>ConfigFile[b][3] or ConfigFile[b][2]<ConfigFile[b][0] or ConfigFile[b][2]>ConfigFile[b][3]:
		print 'Initial Guesses outside bounds for ',b,' :',ConfigFile[b]
		exit()

if isinstance(ConfigFile['RA'], str) or isinstance(ConfigFile['DEC'], str):
	position = SkyCoord(ra=ConfigFile['RA'],dec=ConfigFile['DEC'], unit=(u.hourangle, u.deg))
else:
	position = SkyCoord(ra=ConfigFile['RA'],dec=ConfigFile['DEC'], unit=(u.deg, u.deg))
try:
	hdu = fits.open(ConfigFile['File'])
	size = u.Quantity((ConfigFile['BoxSize'], ConfigFile['BoxSize']), u.arcsec)
	wcs = WCS(hdu[0].header)
	wcs = wcs.sub(2)
	cutout = Cutout2D(hdu[0].data[0][0], position, size, wcs=wcs,copy=True)
	newdata = cutout.data
	newwcs = cutout.wcs
	for i in range(len(newwcs.wcs.crpix)):
		hdu[0].header.set('CRPIX'+str(i+1), newwcs.wcs.crpix[i])
		hdu[0].header.set('CRVAL'+str(i+1), newwcs.wcs.crval[i])

	hdu[0].data = [[newdata]]
	hdu.writeto('Crop.fits', overwrite=True,output_verify='fix')
	ConfigFile['File'] = 'Crop.fits'
except:
	print 'Trimming failed....'

hdu = fits.open(ConfigFile['File'])
wcs = WCS(hdu[0].header)
# print wcs
aux = wcs.all_world2pix([[position.ra.deg,position.dec.deg,0,0]], 1,ra_dec_order=True)
pix_x = aux[0][0]
pix_y = aux[0][1]

# exit()
measured_flux = fits.open(ConfigFile['File'])[0].data[0][0]
Kernel2Use,pix_size,PixelsPerBeam = GetKernel(ConfigFile['File'])
sigma = np.ones_like(measured_flux)*ConfigFile['SigmaValue']/1000.0
sigma = sigma*np.sqrt(PixelsPerBeam)
nburn = int(0.5*ConfigFile['nsteps'])
starting_guesses = []

ConfigFile['x0Bounds'] = 	[
							max(pix_x+ConfigFile['x0Bounds'][0]/pix_size,pix_x-ConfigFile['BoxSize']*0.5/pix_size),
							pix_x+ConfigFile['x0Bounds'][1]/pix_size,
							pix_x+ConfigFile['x0Bounds'][2]/pix_size,
							min(pix_x+ConfigFile['x0Bounds'][3]/pix_size,pix_x+ConfigFile['BoxSize']*0.5/pix_size)
							]

ConfigFile['y0Bounds'] = 	[
							max(pix_y+ConfigFile['y0Bounds'][0]/pix_size,pix_y-ConfigFile['BoxSize']*0.5/pix_size),
							pix_y+ConfigFile['y0Bounds'][1]/pix_size,
							pix_y+ConfigFile['y0Bounds'][2]/pix_size,
							min(pix_y+ConfigFile['y0Bounds'][3]/pix_size,pix_y+ConfigFile['BoxSize']*0.5/pix_size)
							]
for i in range(ConfigFile['nwalkers']):
    aux = 	[
    		np.random.uniform(ConfigFile['AmplitudeBounds'][1],ConfigFile['AmplitudeBounds'][2]),
    		np.random.uniform(ConfigFile['ReffBounds'][1],ConfigFile['ReffBounds'][2]),
    		np.random.uniform(ConfigFile['nBounds'][1],ConfigFile['nBounds'][2]),
    		np.random.uniform(ConfigFile['x0Bounds'][1],ConfigFile['x0Bounds'][2]),
    		np.random.uniform(ConfigFile['y0Bounds'][1],ConfigFile['y0Bounds'][2]),
    		np.random.uniform(ConfigFile['ellipBounds'][1],ConfigFile['ellipBounds'][2]),
    		np.random.uniform(ConfigFile['thetaBounds'][1],ConfigFile['thetaBounds'][2])
    		]
    starting_guesses.append(np.array(aux))
starting_guesses = np.array(starting_guesses)   


print 'Number of iterations:',ConfigFile['ndim']*ConfigFile['nwalkers']*ConfigFile['nsteps']
sampler = emcee.EnsembleSampler(ConfigFile['nwalkers'], ConfigFile['ndim'], lnprob, args=[measured_flux,sigma], threads=ConfigFile['Nthreads'],a=ConfigFile['MCMCA'])
currenttime = time.time()
Step = 1
for pos, prob, state in sampler.sample(starting_guesses, iterations=ConfigFile['nsteps']):
	print 'Step:',Step,'/',ConfigFile['nsteps']
	print "Mean acceptance fraction: %f"%(numpy.mean(sampler.acceptance_fraction))
	print "Mean lnprob and Max lnprob values: %f %f"%(numpy.mean(prob), numpy.max(prob))
	print "Time to run previous set of walkers (seconds): %f"%(time.time() - currenttime)
	currenttime = time.time()
	Step += 1

print '*** Done Fitting... ***'
ll = ['Amp','Reff','n','x0','y0','ellip','PA']

# sampler.run_mcmc(starting_guesses, nsteps)
emcee_trace = sampler.chain[:, :, :].reshape((-1, ConfigFile['ndim']))
lnprob = sampler.lnprobability
print 50*'#'
print '*** Best fit ***'
for i in range(len(ll)):
	if i==3:
		print ll[i],':',(emcee_trace[np.argmax(lnprob)][i]-pix_x)*pix_size
	elif i==4:
		print ll[i],':',(emcee_trace[np.argmax(lnprob)][i]-pix_y)*pix_size
	else:
		print ll[i],':',emcee_trace[np.argmax(lnprob)][i]
# print 'best fit:',emcee_trace[np.argmax(lnprob)]
theta = emcee_trace[np.argmax(lnprob)]
out = fits.open(ConfigFile['File'])
y_model = Sersic2D(measured_flux, *theta)
out[0].data[0][0] = measured_flux - y_model
print 50*'#'
print '*** Sigma Image ***'
print 'Residual Sigma:',format(np.std(measured_flux - y_model)*1000.0,'.2e'),'mJy/beam'
print 'Initial Sigma:',format(ConfigFile['SigmaValue'],'.2e'),'mJy/beam'
out.writeto('Residual.fits',overwrite=True)

out = fits.open(ConfigFile['File'])
y_model = Sersic2D(measured_flux, *theta)
out[0].data[0][0] = y_model
out.writeto('ModelConvolved.fits',overwrite=True)

out = fits.open(ConfigFile['File'])
y_model = Sersic2DWithoutConvolving(measured_flux, *theta)
out[0].data[0][0] = y_model
out.writeto('Model.fits',overwrite=True)

print 50*'#'
print '*** Plotting Traces... ***'
x = np.array([])
y = np.array([])
maxlnprob = np.max(lnprob)
for i in range(len(lnprob)):
    x = np.append(x,range(len(lnprob[i])))
    y = np.append(y,maxlnprob - lnprob[i])
plt.figure()
plt.hexbin(x[y>0],y[y>0], gridsize=[70,30], cmap='inferno',bins='log',mincnt=1,yscale='log',linewidths=0)
plt.ylabel('maxlnprob -lnprob')
plt.xlabel('iteration')
try:
	plt.xlim(min(x),max(x))
	plt.ylim(min(y),max(y))
except:
	print 'Negative values in Convergence....'
plt.savefig('Convergence.png',dpi=300)
plt.close()

for ID in range(ConfigFile['ndim']):
    plt.figure()
    x = np.array([])
    y = np.array([])
    for i in sampler.chain:
    	x = np.append(x,range(len(i.T[ID])))
    	y = np.append(y,i.T[ID])

    plt.figure()
    if (max(y)/min(y))>50 and len(y[y<0])<1:
    	plt.hexbin(x,y, gridsize=[70,30], cmap='inferno',bins='log',mincnt=1,yscale='log',linewidths=0)
    else:
    	plt.hexbin(x,y, gridsize=[70,30], cmap='inferno',bins='log',mincnt=1,linewidths=0)
    plt.ylabel(ll[ID])
    plt.xlabel('iteration')
    plt.xlim(min(x),max(x))
    plt.ylim(min(y),max(y))
    plt.savefig(ll[ID]+'_trace.png',dpi=300)
    plt.close()

print 50*'#'
print '*** Acceptance Fraction ***'
af = sampler.acceptance_fraction
af_msg = '''As a rule of thumb, the acceptance fraction (af) should be 
                          between 0.2 and 0.5
          If af < 0.2 decrease the a parameter
          If af > 0.5 increase the a parameter
          '''
print "Mean acceptance fraction:", np.mean(af)
if np.mean(af)<0.2 or np.mean(af)>0.5:
	print af_msg


samples = sampler.chain[:, nburn:, :].reshape((-1, ConfigFile['ndim']))
lnprob_aux = sampler.lnprobability[:, nburn:].reshape(-1)
print 50*'#'
print '*** Pruning... ***'
try:
	samples,lnprob2 = prune(samples,lnprob_aux)
except:
	print 'Prunning failed....'

rcParams.update({'figure.autolayout': False})
print 50*'#'
print '*** Plotting Covariance... ***'
fig = corner.corner(samples, labels=ll,title_kwargs={'y':1.05},title_fmt=".2f",use_math_text=True,bins=15,quantiles=[0.16, 0.5, 0.84],show_titles=True,color='DarkOrange',hist_kwargs={'color':'black','linewidth':1.5},contour_kwargs={'linewidths':1,'colors':'black'})
fig.savefig("Covariance.pdf")
print 50*'#'
print '*** Posterior parameters and percentiles [16,50,84]***'
for ID in range(ConfigFile['ndim']):
  pc = np.percentile(samples.T[ID], [16,50,84])
  if ID==3:
  	print 'DRA:',round((pc[1]-pix_x)*pix_size,4),'+/-',round(np.mean([(pc[2]-pc[1])*pix_size,(pc[1]-pc[0])])*pix_size,4),(pc - pix_x)*pix_size
  elif ID==4:
  	print 'DDEC:',round((pc[1]-pix_y)*pix_size,4),'+/-',round(np.mean([(pc[2]-pc[1])*pix_size,(pc[1]-pc[0])])*pix_size,4),(pc - pix_y)*pix_size
  else:
  	print ll[ID]+':',round(pc[1],4),'+/-',round(np.mean([pc[2]-pc[1],pc[1]-pc[0]]),4),pc

