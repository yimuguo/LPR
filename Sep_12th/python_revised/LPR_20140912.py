#@setlocal enableextensions & python -x %~f0 %* & goto :EOF
# ---uncomment the above line to make it directly executable by WinNT
# ---but...it may be preferable to run this from a .bat file as:
# ---  c:\python24\python c:\python24\{this program}.py %1 %2 %3 %4 %5 %6

# Long Period Record processor
# This program takes a long sequence of clock periods (like what comes from a TEK7404 JIT3
# package) and generates useful information about the clock such as:
# --- min/mean/max period
# --- period jitter
# --- cycle-to-cycle jitter
# --- Rambus-type 1-6 period group-to-group jitter
# --- peak-to-peak frequency modulation
# --- eye-closure in PCIexpress Gen1, Gen2, and FB-DIMM environments
# --- "spectral purity" (i.e. TEK JIT3 compatible period FFT spectrum)

#--------change these when revising the program---------------
thisProgram = 'LPR_20140912.py'
revLevel = '5.0'
revDate = '2014-04-08'   #JD 20140408
revDate = '2014-09-12'   #AP 20140912 September 12th 2014
#-------------------------------------------------------------

# bring in basic modules
print ('....importing modules')
import os, sys, time

filename = sys.argv[1]
##filename = '498_Cpu0_100_SpOn_1_50mV.txt'
##filename = 'e0_hcsl100_son_2375mv_85c_50mv.txt'
##filename = '6QUA30212A_Agilent81130A_OUT1_142p5MHz_1800mV_Vdd=1p2V_22Ohms_25C.txt'
##filename = '498_Cpu0_100_SpOn_1_50mV_size_power2.txt'
##filename ='498_NS_sas0_133_SpOff_1_50mV.txt'
##filename = '652C-005_LDR_2_5M_PFD_BW_37k_and_HCSL_on_OUT1_SSoff.txt'  #sys.argv[1]
##filename = '183-305_SRC3_SSon_cpu266.txt'  #sys.argv[1]
##filename = 'Wilbur 652C-005_LDR_Default_and_HCSL_on_OUT1_SSon.txt'
##filename = '183-305_SRC2_SSon_cpu266.txt'
##filename = '932SQ420_CPU_SSoff.txt'
##filename = '932SQ420_CPU_SSon.txt'
##filename = '932SQ420_NS_SAS.txt'

customMode = False

try:
   if sys.argv[2] == 'custom':
      customMode = True
except IndexError:
   pass

ltime=time.clock()

def timestamp(s):    # used for program timing
   global ltime
   now=time.clock()
   #print '<<%s>> delta time:'%s, '%.3f'%(now-ltime), 'timestamp:', '%.3f'%now
   print('time: ' + '%5.2f'%now + s)
   ltime=now
#   sys.stdout.flush()  #JD 20140408

timestamp('....starting timer')

from numpy import pi, zeros, ones, array, mean, sqrt, log, \
                  exp, cos, cumsum, convolve, arange, cov, \
                  transpose, shape, hanning, argsort, std , concatenate

from numpy.fft import rfft, irfft

try:
   import PyPy
   PyPy.full()
   print('....PyPy found')
except ImportError:
   pass

#------screen header-------------------------------
print ('Long Period Record (LPR) post processor')
print ('by: Dan Clementi, Integrated Device Technology')
print ('    questions???? email : dan.clementi@idt.com')
print ('    Modified by Aras Pirbadian email : Aras.Pirbadian@gmail.com')
print ('revision ' + revLevel + ' : ' + revDate)
print ('-------------------------')
print ('We report.....you decide!')
print ('-------------------------')

#---extract filename prefix and create output file names-----
splitFilename = os.path.split(filename)
fileNameShort = splitFilename[1]
pathName = os.getcwd()
plot1FileName  = fileNameShort + '.pg1.png'
plot2FileName  = fileNameShort + '.pg2.png'
plot3FileName  = fileNameShort + '.pg3.png'
plot4FileName  = fileNameShort + '.pg4.png'
plot5FileName  = fileNameShort + '.pg5.png'
plot6FileName  = fileNameShort + '.pg6.png'
plot7FileName  = fileNameShort + '.pg7.png'
plot8FileName  = fileNameShort + '.pg8.png'
plot9FileName  = fileNameShort + '.pg9.png'
resultFileName = fileNameShort + '.csv'
#-------------------------------------------
#------Maxima , Minima Calculations On or Off
maxima_cal=sys.argv[3]
maxima_calc=0
minima_calc=0
if maxima_cal == '1':
   maxima_calc=1
   minima_calc=1
#--------------------------------------------
#------Plot Spectrum and Eye Closure ? 1= Yes
spectrum_plot=0
Eye_Closure_plot=0

#------useful constants---------------------
us = 1e-6 ; ns = 1e-9 ; ps = 1e-12
KHz = 1e3 ; MHz = 1e6
#-------------------------------------------
#JD 20140409
def parse(s):
	i = 0
	while (len(s) > 0 and i < len(s)):
		if s[i] == ',':
			s=s[i+1:]
			i=0
		else: i+=1
	return s
#----------------------------------------------------------------
#     This will extract the numbers from a file (one number per line)
#     and ignore lines that aren't floating point numbers.
def readNumbersOnly(filename,NMax):
    n = 0
    vs = zeros(NMax,'d')
    textfile = open(filename,'r')
    for line in textfile:
        if line[0] == ';':continue  #JD 20140409 ignore comments
        line = parse(line)          #JD 20140409 1st column if more than 1
        #print line
        try:
            vs[n]+=float(line)
            n+=1
        except ValueError:
            pass
        except IndexError:
            print ('more than "NMax" data points in file')
            print ('truncating to ' + '%6u' % NMax + ' points')
            return vs
    textfile.close()
    return vs[0:n]
#----------------------------------------------------------------

#----------general interpolation function---------
#          given two points on a line: (x1,y1) and (x2,y2) and a desired new x-value (XVal)
#          will return the y-value on the same line
def returnYVal(XVal,x1,y1,x2,y2):
    newy = (XVal - x1) * ((y2-y1)/(x2-x1)) + y1
    return newy

#--------detrend a sequence (subtract B.S.L. fit)-------
def detrend_linear(x):
    "adapted from matplotlib function"
    xx = arange(float(len(x)))
    X = (array([xx]+[x]))
    C = cov(X)
    b = C[0,1]/C[0,0]
    a = x.mean() - b*xx.mean()
    return x-(b*xx+a)

#------------make a tapered rectangular window function------------------
def cosTaper(N,alpha):        # N is the length of the window, alpha is the fraction of the record
                              # that will be devoted to the cosine fit: (window=1 for N*(1-alpha) samples
    halfCos = int(alpha * N / 2.0)
    temp = array(ones(N,'d'))
    for k in range(0,halfCos):
        temp[k] = 0.5 * (1.0 - cos((2 * pi / (alpha * N + 1.0)) * (k + 1.0)))
        temp[N-1-k] = temp[k]
    return temp

#----------class definition for LPR (Long Period Record)---------------------
#          this contains the data structures and methods used on that data
class LPR:
   def __init__(self, longPeriodRecord):
      self.period = array(longPeriodRecord)
      self.calcPeriodMetrics()
      self.fMean = 1.0 / self.tMean
      self.makeGrids()
   def calcPeriodMetrics(self):
      self.tMin      = self.period.min()
      self.tMinDisp  = (self.tMin/ns, 'Period : min. (ns)')
      self.tMean     = self.period.mean()
      self.tMeanDisp = (self.tMean/ns, 'Period : mean (ns)')
      self.tMax      = self.period.max()
      self.tMaxDisp  = (self.tMax/ns, 'Period : max. (ns)')
      self.pj        = self.period - self.tMean
      self.p2p       = self.tMax - self.tMin
      self.p2pDisp   = (self.p2p/ps, 'Period Jitter : pk-pk  (ps)')
      self.rms       = std(self.pj)
      self.rmsDisp   = (self.rms/ps, 'Period Jitter : R.M.S. (ps)')
      self.c2c       = self.calcC2CJitter()
      self.c2cDisp   = (self.c2c/ps, 'Cycle-to-Cycle Jitter  (ps)')
      self.meanc2c       = self.calcC2CJittermean()
   def calcC2CJitter(self):
      c2cArray = zeros(N-1,'d')
      for n in xrange(1,N):
         c2cArray[n-1] = self.pj[n]-self.pj[n-1]
      return abs(c2cArray).max()
   def calcC2CJittermean(self):
      c2cArray = zeros(N-1,'d')
      for n in xrange(1,N):
         c2cArray[n-1] = self.pj[n]-self.pj[n-1]
      return abs(c2cArray).mean()
   def calcRambusC2C(self, startingCycles, endingCycles):
      self.RambusList = []
      for numCycles in range(startingCycles,endingCycles+1):
         c = []
         for offset in range(0,numCycles):
            start = 2*numCycles+offset
            for k in range(start,N,numCycles):
               c.append(self.edgeTime[k] - self.edgeTime[k-numCycles] - self.edgeTime[k-numCycles] + self.edgeTime[k-2*numCycles])
         ac = array(c)
         absac = abs(ac)
         self.RambusList.append(absac.max())
   def boxcarSmoothedFrequency(self, boxcarLength):
      boxcar = ones(boxcarLength)
      return (boxcarLength/convolve(array(self.period), boxcar)[boxcarLength:-boxcarLength])
   def calcPhaseError(self):
      self.edgeTime = cumsum(self.period)
      self.phi = cumsum(self.pj)
      self.phi = self.phi - self.phi.mean()
   def LTJ(self,boxLength):
      resultLength = len(self.phi)-boxLength
      if resultLength < 0:
         pass
      d = zeros(resultLength,'d')
      for i in range(0,resultLength-1):
         d[i] = self.phi[i+boxLength]-self.phi[i]

      return d.max() - d.min()
   def calculateMeanFrequencyWithSSCGBasedWindow(self):
      #-----special calculation of mean frequency over a whole number of SSCG cycles------
      fSSCG_Nominal = 14.3181818e6 / 440.0
      tSSCG_Nominal = 1.0/fSSCG_Nominal
      totalTime = self.edgeTime[-1]
      N_meanFreqCalc = int(int(totalTime/tSSCG_Nominal) / (totalTime/tSSCG_Nominal) * N + 0.5)
      self.meanFreq_SSCGMult = 1.0 / self.period[0:N_meanFreqCalc].mean()
   def makeGrids(self):
      self.Res = self.fMean / N
      self.freqGrid = array([n * self.Res for n in range(0,N/2+1)])
      self.sGrid = 2 * pi * 1j * self.freqGrid
   def calculateSmoothedFrequency(self):
      self.nFreqDisplay = 16
      self.smoothedFreq16 = self.boxcarSmoothedFrequency(self.nFreqDisplay)
      self.maxSmoothedFreq16 = self.smoothedFreq16.max()
      self.minSmoothedFreq16 = (self.smoothedFreq16).min()
      self.SSCGdeviationPct16 = (self.maxSmoothedFreq16-self.minSmoothedFreq16)/self.fMean*100

      tBoxcar = 1*us
      nBox = int((tBoxcar/(2*self.tMean))+0.5)*2   # figure out the boxLength (in samples), making it an even number
      self.smoothedFreq1u = self.boxcarSmoothedFrequency(nBox)
      self.maxSmoothedFreq1u = self.smoothedFreq1u.max()
      self.minSmoothedFreq1u = self.smoothedFreq1u.min()
      self.SSCGdeviationPct1u = (self.maxSmoothedFreq1u-self.minSmoothedFreq1u)/self.fMean*100

      f3dB_LPF = 2*MHz
      w3dB_LPF = 2 * pi * f3dB_LPF
      af = 0.7071 * w3dB_LPF
      bf = 0.7071 * w3dB_LPF
      def H2MHz(s):
         return((af**2)+(bf**2))/(s**2+(2*s*af)+af**2+bf**2)

      alpha = 0.1
      periodWindow = cosTaper(N,alpha)
      windowedPJ = self.pj * periodWindow     # apply the window to the raw period jitter

      PER = rfft(windowedPJ)   # note that we take the fft of the WINDOWED data
      rawPJSpectrum = PER

      filt2MHzResponse = H2MHz(self.sGrid)

      filteredPJSpectrum = filt2MHzResponse * rawPJSpectrum
      filteredPJ = irfft(filteredPJSpectrum)
      filteredFreq = 1 / (filteredPJ + self.tMean)
      self.maxSmoothedFreq2MHz = filteredFreq.max()
      self.minSmoothedFreq2MHz = filteredFreq.min()

   def calculateLTJMetrics(self):
      #-----------compute 10us long-term jitter--------------------
      nBox_LTJ_10u = int((10e-6/self.tMean)+0.5)
      self.LTJ_10u  = self.LTJ(nBox_LTJ_10u)

      #-----------compute range of long-term jitter values----------
      self.testDelay = [100e-9,300e-9,1e-6,3e-6,10e-6,30e-6,100e-6,300e-6]
      self.LTJResult = []
      for delay in self.testDelay:
         if delay > (0.75 * self.edgeTime[-1]):
            self.testDelay.remove(delay)
         else:
            boxLength = int((delay/self.tMean)+0.5)
            self.LTJResult.append(self.LTJ(boxLength))

   def calcPeriodJitterFFT(self):

      timestamp('.......phase error')
      timeError = detrend_linear(array(self.edgeTime))# detrend to calculate position error (time)
      phaseError = timeError/self.tMean               # convert time error to fractional periods

      timestamp('.......upsampling')
      minUpsample = 4.0                               # calculate the upsampling factor
      newN = 2**(int(log(N*minUpsample)/log(2))+1)    # so that the new number of samples
      upsampleFactor = (float(newN) / float(N))       # is a power of two (note: factor is non-integer)
      edgeLocationUp = array(zeros(N,'d'))
      for i in range(0,N):                            # move the samples to the closest location on
           newLocation = (i + phaseError[i])*upsampleFactor  # the upsampled grid
           edgeLocationUp[i] = int(newLocation)
      edgeLocationUp = edgeLocationUp - edgeLocationUp[0]

      timestamp('.......interpolating')
      upsampledPJ = array(zeros(int(newN*1.3),'d'))   # we make the array bigger than needed
      for j in xrange(1,N):                           # to allow for phase error moving the
          x1 = edgeLocationUp[j-1]          # position out of range
          x2 = edgeLocationUp[j]
          y1 = self.pj[j-1]
          y2 = self.pj[j]
          for k in xrange(int(x1)+1,int(x2)+1):
            upsampledPJ[k] = returnYVal(k,x1,y1,x2,y2)
      upsampledPJ = upsampledPJ[0:newN]               # finally truncate it to proper length (pwr of 2)

      timestamp('.......windowing')
      windowFunc = hanning(newN)                      # TEK uses a hanning window
      self.windowedUpsampledData = upsampledPJ * windowFunc

      newN = len(self.windowedUpsampledData)
      timestamp('.......taking FFT of upsampled period jitter')
      ND = 2.0*sqrt(2.0)*abs(rfft(self.windowedUpsampledData))/newN
      self.subND = ND[0:N/2+1]

   def extractSpursFromPJFFT(self,numPeaks):
      #-------------extract worst spurs-----------------------------------
      self.numPeaks = numPeaks
      self.MaxAmpl = []
      self.AssocFr = []
      AA = argsort(self.subND)             # sort on spur amplitude (argsort returns the array index of smallest to largest)
      for i in range(len(AA)-1,len(AA)-1-self.numPeaks,-1):
          self.MaxAmpl.append(self.subND[AA[i]])
          self.AssocFr.append(self.freqGrid[AA[i]])
      self.MaxAmpl_rs = []
      self.AssocFr_rs = []
      BB = argsort(self.AssocFr)           # resort those top spurs by frequency
      for i in range(0,len(BB)):
          self.MaxAmpl_rs.append(self.MaxAmpl[BB[i]])
          self.AssocFr_rs.append(self.AssocFr[BB[i]])
   def maxima(self):                       # to calculate mean max and min of the local maximums
      number_of_maxes=len(self.smoothedFreq16)/1000 #estimate the length needed and pre assign the arrays
      max_local=[0]*number_of_maxes
      index_max_local=[0]*number_of_maxes
      mean_freq=mean(lpr1.smoothedFreq16)
      max_counter=0
      pos_neg=1
      i=0
      while i in range(len(self.smoothedFreq16)):
          if pos_neg ==1:                 #to look for the first larger than mean number
              if (self.smoothedFreq16[i]-mean_freq) >0:
                  low=i
                  i=i+1000                # Assuming a minimum period of 2000, not all the values need to be searched
                  pos_neg=0
          else:                           #to look for the first less than mean number
              if (self.smoothedFreq16[i]-mean_freq) <0:
                  high=i
                  i=i+1000
                  pos_neg=1            
                  max_array=0
                  for j in range(low+300,high-300): #find the maximum in the positive range 
                     if self.smoothedFreq16[j] > max_array:
                         max_array = self.smoothedFreq16[j]
                         max_Index = j
                  max_local[max_counter]=max_array
                  index_max_local[max_counter]=max_Index            
                  max_counter+=1
                  if max_counter >2:               #Only find the first 3 maximums since this is a slow method to obtain the period
                     break
                  
          i+=1
      period_fix=index_max_local[max_counter-1]-index_max_local[max_counter-2]
      max_counter_start=max_counter
      i=3
      while i in range(max_counter_start,len(self.smoothedFreq16)/period_fix+10 ):     #Estimate the length of array needed based on the known period
         period=index_max_local[max_counter-1]-index_max_local[max_counter-2]
         next_center=index_max_local[max_counter-1]+period
         max_array=0
         if next_center+int(.1*period)>= len(self.smoothedFreq16):         # making sure it won't go over the size
            break
         for j in range(next_center-int(.1*period),next_center+int(.1*period)):
            if self.smoothedFreq16[j] > max_array:
               max_array = self.smoothedFreq16[j]
               max_Index = j   
         max_local[max_counter]=max_array
         index_max_local[max_counter]=max_Index
         max_counter+=1
         i+=1

      self.max_local_all=max_local[1:max_counter]              #cutting the first value since we are not sure if it is a real maximum
      self.index_max_local_all=index_max_local[1:max_counter]

##      if max(self.smoothedFreq16) > max(self.max_local_all):
##         self.max_local_all.append(max(self.smoothedFreq16))
       
      ##   index_max_local_all.append(0)
      ##   for i in range(next_center - period,len(lpr1.smoothedFreq16)):
      ##      if lpr1.smoothedFreq16[i]==max(lpr1.smoothedFreq16):
      ##         index_max_local_all.append(i)
      ##         break
      self.maxima_mean=mean(self.max_local_all)                
      self.maxima_max=max(self.max_local_all)
      self.maxima_min=min(self.max_local_all)

   def minima(self):                       # follow the maxima comments and invert them for minimums. to calculate mean max and min of the local minimums
      number_of_mins=len(self.smoothedFreq16)/1000 #estimate the length needed and pre assign the arrays
      min_local=[0]*number_of_mins
      index_min_local=[0]*number_of_mins
      mean_freq=mean(lpr1.smoothedFreq16)
      min_counter=0
      pos_neg=0
      i=0
      while i in range(len(self.smoothedFreq16)):
          if pos_neg ==1:                 #to look for the first more than mean number
              if (self.smoothedFreq16[i]-mean_freq) >0:
                  high=i
                  i=i+1000                # Assuming a minimum period not all the values need to be searched
                  pos_neg=0
                  for j in range(low+300,high-300): #find the minimum in the positive range 
                     if self.smoothedFreq16[j] < min_array:
                         min_array = self.smoothedFreq16[j]
                         min_Index = j
                  min_local[min_counter]=min_array
                  index_min_local[min_counter]=min_Index            
                  min_counter+=1
                  if min_counter >2:
                     break
          else:                           #to look for the first less than mean number
              if (self.smoothedFreq16[i]-mean_freq) <0:
                  low=i
                  i=i+1000
                  pos_neg=1            
                  min_array=1e12

                  
          i+=1
      period_fix=index_min_local[min_counter-1]-index_min_local[min_counter-2]
      min_counter_start=min_counter
      i=3
      while i in range(min_counter_start,len(self.smoothedFreq16)/period_fix+10 ):
         period=index_min_local[min_counter-1]-index_min_local[min_counter-2]
         next_center=index_min_local[min_counter-1]+period
         min_array=1e12
         if next_center+int(.1*period)>= len(self.smoothedFreq16):
            break
         for j in range(next_center-int(.1*period),next_center+int(.1*period)):
            if self.smoothedFreq16[j] < min_array:
               min_array = self.smoothedFreq16[j]
               min_Index = j   
         min_local[min_counter]=min_array
         index_min_local[min_counter]=min_Index
         min_counter+=1
         i+=1

      self.min_local_all=min_local[1:min_counter]
      self.index_min_local_all=index_min_local[1:min_counter]

##      if min(self.smoothedFreq16) > min(self.min_local_all):
##         self.min_local_all.append(min(self.smoothedFreq16))
       
      ##   index_min_local_all.append(0)
      ##   for i in range(next_center - period,len(lpr1.smoothedFreq16)):
      ##      if lpr1.smoothedFreq16[i]==min(lpr1.smoothedFreq16):
      ##         index_min_local_all.append(i)
      ##         break
      self.minima_mean=mean(self.min_local_all)
      self.minima_max=max(self.min_local_all)
      self.minima_min=min(self.min_local_all)
      
   def calcFreq_period_mean_min_max(self):
      index_max_local_all_period = zeros(len(self.index_max_local_all)-1,'d')
      for n in xrange(1,len(self.index_max_local_all)):
         index_max_local_all_period[n-1] = self.index_max_local_all[n]-self.index_max_local_all[n-1]
      self.Freq_Mod_Period_mean=mean(100000/abs(index_max_local_all_period))  #mean of the 1/T 
      self.Freq_Mod_Period_min=min(100000/abs(index_max_local_all_period))    
      self.Freq_Mod_Period_max=max(100000/abs(index_max_local_all_period))
      
class EC_analysis:
   def __init__(self,name,f3db1,zeta1,f3db2,zeta2, \
                f3db3,fx1,fx2,delay1,delay2,sscgRemoveFlag,responseType,integrationRange):
       self.name = name
       self.f3db1 = f3db1
       self.zeta1 = zeta1
       self.f3db2 = f3db2
       self.zeta2 = zeta2
       self.f3db3 = f3db3
       self.fx1 = fx1
       self.fx2 = fx2
       self.delay1 = delay1
       self.delay2 = delay2
       self.wn1 = (2*pi*self.f3db1 / \
              ((1.0 + 2.0*self.zeta1**2.0 + ((1.0 + 2.0*self.zeta1**2.0)**2.0+1.0)**0.5)**0.5))
       self.wn2 = (2*pi*self.f3db2 / \
              ((1.0 + 2.0*self.zeta2**2.0 + ((1.0 + 2.0*self.zeta2**2.0)**2.0+1.0)**0.5)**0.5))
       self.w3 = (2*pi*self.f3db3)
       self.wx1 = (2*pi*self.fx1)
       self.wx2 = (2*pi*self.fx2)
       self.sscgRemoveFlag = sscgRemoveFlag
       self.responseType = responseType
       self.integrationRange = integrationRange
   def H_commonClocked(self,s):
       H1 = ((2*s*self.zeta1*self.wn1 + self.wn1**2) / \
           (s**2 + 2*s*self.zeta1*self.wn1 + self.wn1**2)) * \
           (1.0 / (1 + s/self.wx1)) * \
           (exp(-self.delay1*s))
       H2 = ((2*s*self.zeta2*self.wn2 + self.wn2**2) / \
           (s**2 + 2*s*self.zeta2*self.wn2 + self.wn2**2)) * \
           (1.0 / (1 + s/self.wx2)) * \
           (exp(-self.delay2*s))
       H3 = (s/(s+self.w3))
       return((H1-H2)*H3)
   def H_dataClocked(self,s):
       H1 = ((2*s*self.zeta1*self.wn1 + self.wn1**2) / \
           (s**2 + 2*s*self.zeta1*self.wn1 + self.wn1**2)) * \
           (1.0 / (1 + s/self.wx1)) * \
           (exp(-self.delay1*s))
       H2 = ((2*s*self.zeta2*self.wn2 + self.wn2**2) / \
           (s**2 + 2*s*self.zeta2*self.wn2 + self.wn2**2)) * \
           (1.0 / (1 + s/self.wx2)) * \
           (exp(-self.delay2*s))
       return(H1*(1-H2))
      
   def H_SRIS(self,s):
       H1 = ((2*s*self.zeta1*self.wn1 + (self.wn1)**2) / \
           (s**2 + 2*s*self.zeta1*self.wn1 + (self.wn1)**2)) 
       H3 = (s**2/(s**2+s*2*pi*10**7+2.2*(2*pi)**2*10**12)* \
             (s**2 + 2*s*2*pi*10**7+(2*pi*10**7)**2)/ \
             (s**2 + 2*s/sqrt(2)*2*pi*10**7+(2*pi*10**7)**2))
       return(H1*H3)
      
   def H_SRIS_GEN2(self,s):
       H1 = ((2*s*self.zeta1*self.wn1 + self.wn1**2) / \
           (s**2 + 2*s*self.zeta1*self.wn1 + self.wn1**2))
       H3 = ((s**2) / \
           (s**2 + sqrt(2)*s*self.w3+ self.w3**2))
##       H3 = (s**2/(s**2+s*2*pi*10**7+2.2*(2*pi)**2*10**12)* \
##             (s**2 + 2*s*2*pi*10**7+(2*pi*10**7)**2)/ \
##             (s**2 + 2*s/sqrt(2)*2*pi*10**7+(2*pi*10**7)**2))
##       H3 = (s/(s+self.w3))
       return(H1*H3)

   def calcFilterResponse(self):
      if (self.responseType == 'COMMONCLOCKED'):
         self.filterResponse = self.H_commonClocked(lpr1.sGrid)
      if (self.responseType == 'DATACLOCKED'):
         self.filterResponse = self.H_dataClocked(lpr1.sGrid)
      if (self.responseType == 'SRIS'):
         self.filterResponse = self.H_SRIS(lpr1.sGrid)
      if (self.responseType == 'SRIS_GEN2'):
         self.filterResponse = self.H_SRIS_GEN2(lpr1.sGrid)
   def calcFilteredPhaseSpectrum(self,rawPhaseSpectrum):
      self.calcFilterResponse()
      self.filteredPhaseSpectrum =  self.filterResponse * rawPhaseSpectrum
   def calcEyeClosureTrend(self):
      ''' converts the DSB spectrum to a real-valued eye closure '''
      lowIndex = int((self.integrationRange[0]/lpr1.fMean)*N+0.5)
      if self.integrationRange[1] == "max":
         highIndex = N/2+1
      else:
         highIndex = int((self.integrationRange[1]/lpr1.fMean)*N)
      filter = zeros(N/2+1,'f')
      for i in range(lowIndex,highIndex):
         filter[i] = 1.0
      self.BPFfilteredPhaseSpectrum = self.filteredPhaseSpectrum * filter 
      self.eyeClosureTrend = irfft(self.BPFfilteredPhaseSpectrum+mean(self.BPFfilteredPhaseSpectrum)) * N 

   def calc_P2P_RMS(self, lowerLim, upperLim):
      ''' returns peak-to-peak and mean from the eye-closure trend '''
      self.rms = std(self.eyeClosureTrend[lowerLim:upperLim])
      self.p2p = self.eyeClosureTrend[lowerLim:upperLim].max() - \
                 self.eyeClosureTrend[lowerLim:upperLim].min()
      return(self.p2p, self.rms)
   def calcCumMSPhaseSpectrum(self,spectrum):
      self.cumMSPhaseSpectrum = zeros(N/2+1,'d')
      for i in range(1,N/2+1):
         self.cumMSPhaseSpectrum[i] = (self.cumMSPhaseSpectrum[i-1] + abs(spectrum[i])**2)
      return(self.cumMSPhaseSpectrum)
   def calc_RMS_Freq(self):
      ''' Calculate RMS in Freq domain '''
      self.cum_rms=zeros(N/2+1,'d')
      for i in range(1,N/2+1):
         self.cum_rms[i] = self.cum_rms[i-1] + abs(self.BPFfilteredPhaseSpectrum[i])**2
      self.cum_rms_by_n2=self.cum_rms[i]
      self.rms_freq=sqrt(self.cum_rms_by_n2*2)
##      self.rms_freq = self.cum_rms_sqrt 
      return(self.rms_freq)

#-----------read in the long period record-----------------------
NMax = 750000                                 # set a maximum record length
timestamp('....reading file: ' + fileNameShort)
rawData = readNumbersOnly(filename,NMax)      # read the data
timestamp('....done reading file')
#----------------------------------------------------------------

###-----------discard non-period data and adjust length------------
if rawData[0]>1e-3 or rawData[1]>1e-3:  # discard the first two numerical points if
    rawData = rawData[2:len(rawData)]   # they don't look like period data
##N = (len(rawData)/2**10) * 2**10        # we make use of as much data as exists while
                                        # making the FFT go faster by discarding up to 1024 points
###-----------------------------------------------------------------
N = (len(rawData))
# ----- instantiate the long period record with the data ---------
lpr1 = LPR(rawData[0:N])
##size_adjed=concatenate([rawData[0:N], rawData[0:219]])
##lpr1 = LPR(size_adjed)
##N=1024*122

# ---- call the calculation routines and work on the LPR ----------------
timestamp('....period jitter calculations')
lpr1.calcPeriodMetrics()

timestamp('....phase jitter calculations')
lpr1.calcPhaseError()

timestamp('....block jitter calculations')
minRambusCycles = 1
maxRambusCycles = 6
lpr1.calcRambusC2C(minRambusCycles,maxRambusCycles)

timestamp('....smoothed frequency calculations')
lpr1.calculateMeanFrequencyWithSSCGBasedWindow()
lpr1.calculateSmoothedFrequency()
if maxima_calc==1:
   timestamp('....Maxima calculations')
   lpr1.maxima()
   lpr1.calcFreq_period_mean_min_max()
if minima_calc==1:
   timestamp('....Minima calculations')
   lpr1.minima()

timestamp('....long-term jitter calculations')
lpr1.calculateLTJMetrics()

timestamp('....period jitter FFT calculations')
lpr1.calcPeriodJitterFFT()
numPks = 40
lpr1.extractSpursFromPJFFT(numPks)
#------------------------------------------------------------------------

def index(freq):
    return int(freq/lpr1.Res + 0.5)

def returnMaxValIndex(realArray, startIndex, endIndex):
    maxResult = 0
    maxIndex = -1
    evalRange = range(startIndex,endIndex+1)
    for i in evalRange:
        if realArray[i] > maxResult:
            maxResult = realArray[i]
            maxIndex = i
    return maxIndex

#-----------------------------------------------------------------------------------
#----This is the main class with the data structure and
#----methods for eye closure !!!

#------Instantiate each of the eye closure analyses
#      The sequence is: name,f3db1,zeta1,f3db2,zeta2,f3db3,
#                       fx1,fx2,delay1,delay2,sscgRemoveFlag,analysisType
#                       analysisType must be either 'COMMONCLOCKED' to use normal differencing
#                       functions for systems with distributed clock or 'DATACLOCKED' for
#                       systems that recover the clock from data (e.g. SATA-II, PCIe Gen2 alternate)

fullRange = (0, "max")
gen2LoRange = (10e3, 1.5e6)
gen2HiRange = (1.5e6, "max")

PCIe_1               = EC_analysis('Gen1 E.C.', \
                       22e6,0.54,1.5e6,0.54,1.5e6,200e6,200e6,0e-9,10e-9,False,'COMMONCLOCKED',fullRange)
PCIe_2A_COMMON_Low   = EC_analysis('Gen2A Low', \
                       16e6,0.54,5e6,1.16,1.0e0,2000e6,2000e6,0e-9,12e-9,True,'COMMONCLOCKED',gen2LoRange)
PCIe_2A_COMMON_High  = EC_analysis('Gen2A High', \
                       16e6,0.54,5e6,1.16,1.0e0,2000e6,2000e6,0e-9,12e-9,True,'COMMONCLOCKED',gen2HiRange)
PCIe_2B_COMMON_Low   = EC_analysis('Gen2B Low', \
                       16e6,0.54,8e6,0.54,1.0e0,2000e6,2000e6,0e-9,12e-9,True,'COMMONCLOCKED',gen2LoRange)
PCIe_2B_COMMON_High  = EC_analysis('Gen2B High', \
                       16e6,0.54,8e6,0.54,1.0e0,2000e6,2000e6,0e-9,12e-9,True,'COMMONCLOCKED',gen2HiRange)
PCIe_2_SRIS   = EC_analysis('Gen2 SRIS', \
                       16e6,5.4e-1,0,0,4.8586e6,2.0e11,2.e11,0e-8,1.2e-8,True,'SRIS_GEN2',fullRange)

PCIe_Gen3_COMMON_2_2_4_2    = EC_analysis('Gen3 4MHz', \
                       2.000e6,7.3e-1,4.00e6,7.3e-1,1.0e7,2.0e15,2.e15,1.2e-8,0e-8,True,'COMMONCLOCKED',fullRange)

PCIe_Gen3_COMMON_2_1_5_1   = EC_analysis('Gen3 5MHz', \
                       2.00e6,1.16e0,5.000e6,1.16e0,1.0e7,2.0e15,2.e15,1.2e-8,0e-8,True,'COMMONCLOCKED',fullRange)

PCIe_3_SRIS   = EC_analysis('Gen3 SRIS', \
                       4e6,7.3e-1,0,0,1.0e7,2.0e11,2.e11,0e-8,1.2e-8,True,'SRIS',fullRange)






def removeSSCG(spectrum):
   ''' removes 32KHz (nominal) SSCG and it's odd harmonics up to the 23rd '''
   absSpectrum = abs(spectrum)
   fundSSCGindex = returnMaxValIndex(absSpectrum,index(30e3),index(35e3))
   lookRange = index(15e3)
   squashRange = index(15e3)
   extractedFft = 1.0 * spectrum
   for harmN in range(1,23,2):
      harmCenterIndex = harmN * fundSSCGindex
      if harmN == 1:
         thisSpurIndex = fundSSCGindex
      else:
         thisSpurIndex = returnMaxValIndex(absSpectrum, \
                                          (harmCenterIndex-lookRange), \
                                          (harmCenterIndex+lookRange))
      lowMeanRangeTop = thisSpurIndex - squashRange
      lowMeanRangeBottom = max(lowMeanRangeTop - 10,1)
      highMeanRangeBottom = thisSpurIndex + squashRange
      highMeanRangeTop = highMeanRangeBottom + 10
      lowMeanRangeMean = mean(absSpectrum[lowMeanRangeBottom:lowMeanRangeTop+1])
      highMeanRangeMean = mean(absSpectrum[highMeanRangeBottom:highMeanRangeTop+1])
      for i in range(lowMeanRangeTop,highMeanRangeBottom+1):
         newVal = returnYVal(i,lowMeanRangeTop,lowMeanRangeMean, \
                             highMeanRangeBottom,highMeanRangeMean)
         extractedFft[i] = newVal
      squashRange = index(4e3)
##   return spectrum
   return extractedFft
#----Here is where all the work gets done for eye-closure calculations!--------

timestamp('....computing raw phase jitter FFT')
alpha = 0.3
phaseWindow = cosTaper(N,alpha)

windowedPhaseError = lpr1.phi #* phaseWindow                  # apply the window to the raw phase jitter
windowedPhaseSpectrum = rfft(windowedPhaseError) / N   # do the FFT to get the spectrum
windowedSSRemovedPhaseSpectrum = removeSSCG(windowedPhaseSpectrum)

# for the RMS and peak to peak calculations, we don't look at the head and tail
# because the window would distort the RMS calculation

firstEval_old = int((alpha / 2.0) * N)
lastEval_old = N - firstEval_old

firstEval_25 =  25 #Dropping 25 samples because thats the length of the side spikes
lastEval_25 = N - firstEval_25

ECspecList = (PCIe_1, \
              PCIe_2A_COMMON_Low, PCIe_2A_COMMON_High, \
              PCIe_2B_COMMON_Low, PCIe_2B_COMMON_High
              )

for spec in ECspecList:
   timestamp('....computing '+ spec.name)
   if spec.sscgRemoveFlag == True:
      spec.calcFilteredPhaseSpectrum(windowedSSRemovedPhaseSpectrum)
   else:
      spec.calcFilteredPhaseSpectrum(windowedPhaseSpectrum)
   spec.calcEyeClosureTrend()
   spec.calc_P2P_RMS(firstEval_old,lastEval_old)
   spec.calc_RMS_Freq()


ECspecList = (PCIe_Gen3_COMMON_2_2_4_2, \
              PCIe_Gen3_COMMON_2_1_5_1, \
              PCIe_2_SRIS, \
              PCIe_3_SRIS
              )

for spec in ECspecList:
   timestamp('....computing '+ spec.name)
   if spec.sscgRemoveFlag == True:
      spec.calcFilteredPhaseSpectrum(windowedSSRemovedPhaseSpectrum)
   else:
      spec.calcFilteredPhaseSpectrum(windowedPhaseSpectrum)
   spec.calcEyeClosureTrend()
   spec.calc_P2P_RMS(firstEval_25,lastEval_25)
   spec.calc_RMS_Freq()




#---------pick the worst of the multiple filters used on PCIe2-----------
#PCIe_2_WORST_p2p = max(PCIe_2A_COMMON.p2p, PCIe_2B_COMMON.p2p)
#PCIe_2_WORST_rms = max(PCIe_2A_COMMON.rms, PCIe_2B_COMMON.rms)
PCIe_2_WORST_rms_loBand = max(PCIe_2A_COMMON_Low.rms, PCIe_2B_COMMON_Low.rms)
PCIe_2_WORST_rms_hiBand = max(PCIe_2A_COMMON_High.rms, PCIe_2B_COMMON_High.rms)

#---------pick the worst of the multiple filters used on Gen3 PCIe3-----------
Gen3_PCIe_3_WORST_p2p = max(PCIe_Gen3_COMMON_2_2_4_2.p2p, PCIe_Gen3_COMMON_2_1_5_1.p2p 
                            )
#Gen3_PCIe_3_WORST_rms = max(PCIe_Gen3_COMMON_2_2_4_2.rms, PCIe_Gen3_COMMON_2_1_5_1.rms )
Gen3_PCIe_3_WORST_rms = max(PCIe_Gen3_COMMON_2_2_4_2.rms, PCIe_Gen3_COMMON_2_1_5_1.rms
                                 )
Gen3_PCIe_3_WORST_rms_freq = max(PCIe_Gen3_COMMON_2_2_4_2.rms_freq, PCIe_Gen3_COMMON_2_1_5_1.rms_freq
                                 )
#---------pick the worst of the multiple filters used on PCIe 2-3 SRIS-----------
PCIe_2_SRIS_WORST_p2p = PCIe_2_SRIS.p2p
PCIe_2_SRIS_WORST_rms = PCIe_2_SRIS.rms
PCIe_2_SRIS_WORST_rms_freq = PCIe_2_SRIS.rms_freq

PCIe_3_SRIS_WORST_p2p = PCIe_3_SRIS.p2p
PCIe_3_SRIS_WORST_rms = PCIe_3_SRIS.rms
PCIe_3_SRIS_WORST_rms_freq = PCIe_3_SRIS.rms_freq

#------------------------------------------------------------------------------

#----------create .csv file for Excel import------------------
timestamp('....creating .csv file with results')
resultFile = open(resultFileName,'w')

str = 'Source File: , ' + fileNameShort
resultFile.write(str + '\n')

str = 'LPR Post-Processor Revision: , ' + \
      revLevel
resultFile.write(str + '\n')

str = 'Frequency : mean (MHz) , ' + \
      '%.3f' % (lpr1.meanFreq_SSCGMult/MHz)
resultFile.write(str + '\n')
print(str)

str = 'Frequency (N=16) : min (MHz) , ' + \
      '%.3f' % (lpr1.minSmoothedFreq16/MHz)
resultFile.write(str + '\n')
print(str)

str = 'Frequency (N=16) : max (MHz) , ' + \
      '%.3f' % (lpr1.maxSmoothedFreq16/MHz)
resultFile.write(str + '\n')
print(str)

str = 'Frequency (1us avg) : min (MHz) , ' + \
      '%.3f' % (lpr1.minSmoothedFreq1u/MHz)
resultFile.write(str + '\n')
print(str)

str = 'Frequency (1us avg) : max (MHz) , ' + \
      '%.3f' % (lpr1.maxSmoothedFreq1u/MHz)
resultFile.write(str + '\n')
print(str)

str = 'Frequency (N=16) : pk-pk (MHz) , ' + \
     '%.3f' % ((lpr1.maxSmoothedFreq16-lpr1.minSmoothedFreq16)/MHz)
resultFile.write(str + '\n')
print(str)

str = 'Frequency (N=16) : pk-pk (%) , ' + \
      '%.3f' % (lpr1.SSCGdeviationPct16)
resultFile.write(str + '\n')
print(str)
if maxima_calc==1:
   str = 'Frequency Mod Maxima Max (MHz), ' + \
         '%.6f' % (lpr1.maxima_max/MHz)
   resultFile.write(str + '\n')
   print(str)

   str = 'Frequency Mod Maxima Mean (MHz), ' + \
         '%.6f' % (lpr1.maxima_mean/MHz)
   resultFile.write(str + '\n')
   print(str)

   str = 'Frequency Mod Maxima Min (MHz), ' + \
         '%.6f' % (lpr1.maxima_min/MHz)
   resultFile.write(str + '\n')
   print(str)

if minima_calc==1:
   str = 'Frequency Mod Minima Max (MHz), ' + \
         '%.6f' % (lpr1.minima_max/MHz)
   resultFile.write(str + '\n')
   print(str)

   str = 'Frequency Mod Minima Mean (MHz), ' + \
         '%.6f' % (lpr1.minima_mean/MHz)
   resultFile.write(str + '\n')
   print(str)

   str = 'Frequency Mod Minima Min (MHz), ' + \
         '%.6f' % (lpr1.minima_min/MHz)
   resultFile.write(str + '\n')
   print(str)

if maxima_calc==1:
   str = 'Frequency Mod Maxima Max (PPM), ' + \
         '%.2f' % ((lpr1.maxima_max-lpr1.maxima_mean)/lpr1.maxima_mean*MHz)
   resultFile.write(str + '\n')
   print(str)

   str = 'Frequency Mod Maxima Mean (PPM), ' + \
         '%.2f' % ((lpr1.maxima_mean-lpr1.maxima_mean)/lpr1.maxima_mean*MHz)
   resultFile.write(str + '\n')
   print(str)

   str = 'Frequency Mod Maxima Min (PPM), ' + \
         '%.2f' % ((lpr1.maxima_min-lpr1.maxima_mean)/lpr1.maxima_mean*MHz)
   resultFile.write(str + '\n')
   print(str)
if minima_calc==1 and maxima_calc==1:
   str = 'Frequency Mod Minima Max (PPM), ' + \
         '%.2f' % ((lpr1.minima_max-lpr1.maxima_mean)/lpr1.maxima_mean*MHz)
   resultFile.write(str + '\n')
   print(str)

   str = 'Frequency Mod Minima Mean (PPM), ' + \
         '%.2f' % ((lpr1.minima_mean-lpr1.maxima_mean)/lpr1.maxima_mean*MHz)
   resultFile.write(str + '\n')
   print(str)

   str = 'Frequency Mod Minima Min (PPM), ' + \
         '%.2f' % ((lpr1.minima_min-lpr1.maxima_mean)/lpr1.maxima_mean*MHz)
   resultFile.write(str + '\n')
   print(str)

   str = 'Frequency Mod Maxima Max - Min (PPM), ' + \
         '%.2f' % ((lpr1.maxima_max-lpr1.maxima_mean)/lpr1.maxima_mean*MHz - (lpr1.maxima_min-lpr1.maxima_mean)/lpr1.maxima_mean*MHz)
   resultFile.write(str + '\n')
   print(str)

   str = 'Frequency Mod Minima Max - Min (PPM), ' + \
         '%.2f' % ((lpr1.minima_max-lpr1.minima_mean)/lpr1.minima_mean*MHz - (lpr1.minima_min-lpr1.minima_mean)/lpr1.minima_mean*MHz)
   resultFile.write(str + '\n')
   print(str)
if maxima_calc==1:
   str = 'Frequency Mod 1/Tp2p Max (KHz), ' + \
         '%.3f' % (lpr1.Freq_Mod_Period_max)
   resultFile.write(str + '\n')
   print(str)

   str = 'Frequency Mod 1/Tp2p Mean (KHz), ' + \
         '%.3f' % (lpr1.Freq_Mod_Period_mean)
   resultFile.write(str + '\n')
   print(str)

   str = 'Frequency Mod 1/Tp2p Min (KHz) , ' + \
         '%.3f' % (lpr1.Freq_Mod_Period_min)
   resultFile.write(str + '\n')
   print(str)

periodSpecList = (lpr1.tMeanDisp, lpr1.tMinDisp, lpr1.tMaxDisp, \
                  lpr1.p2pDisp, lpr1.rmsDisp, lpr1.c2cDisp)
for i in periodSpecList:
    str = i[1] + ',' + '%.3f' % i[0]
    resultFile.write(str + '\n')
    print(str)

str = 'C2C Jitter Mean : (ps), ' + \
    '%.3f' % (lpr1.meanc2c/ps)
resultFile.write(str + '\n')
print(str)

str = 'Long-Term Jitter @ 10us : (ps), ' + \
    '%.3f' % (lpr1.LTJ_10u/ps)
resultFile.write(str + '\n')
print(str)

ECspecList = (PCIe_1, \
              PCIe_2A_COMMON_Low, PCIe_2A_COMMON_High, \
              PCIe_2B_COMMON_Low, PCIe_2B_COMMON_High, \
              PCIe_Gen3_COMMON_2_2_4_2, \
              PCIe_Gen3_COMMON_2_1_5_1, \
              PCIe_2_SRIS, \
              PCIe_3_SRIS
              )

for spec in ECspecList:
    if spec == PCIe_1:
         str = spec.name + ' : pk-pk (ps),' + '%3.3f' % (spec.p2p/ps)
         resultFile.write(str + '\n')
         print(str)
         str = 'PCIe Gen 2 WORST RMS loBand (ps) , ' + \
         '%.3f' % (PCIe_2_WORST_rms_loBand/ps)
         resultFile.write(str + '\n')
         print(str)

         str = 'PCIe Gen 2 WORST RMS hiBand (ps) , ' + \
               '%.3f' % (PCIe_2_WORST_rms_hiBand/ps)
         resultFile.write(str + '\n')
         print(str)

         str = PCIe_2_SRIS.name + ' : RMS   (ps),' + '%3.3f' % (spec.rms/ps)
         resultFile.write(str + '\n')
         print(str)

         str = 'PCIe Gen 3 WORST RMS(ps) , ' + \
               '%.3f' % (Gen3_PCIe_3_WORST_rms/ps)
         resultFile.write(str + '\n')
         print(str)

         str = PCIe_3_SRIS.name + ' : RMS   (ps),' + '%3.3f' % (spec.rms/ps)
         resultFile.write(str + '\n')
         print(str)


    str = spec.name + ' : RMS   (ps),' + '%3.3f' % (spec.rms/ps)
    resultFile.write(str + '\n')
    print(str)
    
for i in range(1,7):
    str = 'C2C Jitter - ' + '%2d' % i + '-cycle (ps):, ' + '%3.3f' % (lpr1.RambusList[i-1]/ps)
    print(str)
    resultFile.write(str + '\n')

str = 'Largest Spurs (by magnitude), Spur Freq. - MHz, Spur Magnitude - ps'
resultFile.write(str + '\n')
for i in range(0,lpr1.numPeaks):
    str = ',' + ('%7.4f' % (lpr1.AssocFr[i]/MHz) + ',' + '%6.3f' % (lpr1.MaxAmpl[i]/ps))
    resultFile.write(str + '\n')

str = 'Largest Spurs (by frequency), Spur Freq. - MHz, Spur Magnitude - ps'
resultFile.write(str + '\n')
for i in range(0,lpr1.numPeaks):
    str = ',' + ('%7.4f' % (lpr1.AssocFr_rs[i]/MHz) + ',' + '%6.3f' % (lpr1.MaxAmpl_rs[i]/ps))
    resultFile.write(str + '\n')

resultFile.close()
#------------------------------------------------------------------------------


#from pylab import            *  #JD 20140408    
from matplotlib.pyplot import *  #JD 20140408

#------------------------------------------------------------------------------
timestamp('....generating time trend plots')

##def createPlotLayout():
##
##   return


def indexToTime(index):
    return index * lpr1.tMean
def timeToIndex(tme):
    return int(tme / lpr1.tMean + 0.5)

dispIndexMin = int((alpha/2.0) * N)
rawGraphStartTime = indexToTime(dispIndexMin)
minXIndex = timeToIndex(rawGraphStartTime)

nSSCGPlotWidth = 20
timeSpanToDisplay = nSSCGPlotWidth / (14.318e6/440)
rawGraphEndTime = rawGraphStartTime + timeSpanToDisplay
maxXIndex = min(timeToIndex(rawGraphEndTime),N-16)

def promptForNewVal(prompt, defaultVal):
    try:
        newVal = input(prompt)
    except SyntaxError:
        newVal = defaultVal
    return newVal

#------------------------------------------------------------

if customMode == True:

   print('default graph start/finish is (in us) : ' + '%.0f' % (rawGraphStartTime/us)+ '/' + '%.0f' % (rawGraphEndTime/us))
   newStartTimeInUs = promptForNewVal('enter new start time (in us), <CR>=default : ', rawGraphStartTime/us)
   newEndTimeInUs = promptForNewVal('enter new finish time (in us), <CR>=default : ', rawGraphEndTime/us)

   rawGraphStartTime = newStartTimeInUs * us
   minXIndex = timeToIndex(rawGraphStartTime)
   rawGraphEndTime = newEndTimeInUs * us
   maxXIndex = timeToIndex(rawGraphEndTime)


xLabelSize = 7
yLabelSize = 7
labelTextSize = 8
resultSize = 6
resultFont = 'Courier'

figure(figsize=(10.24,7.68))

plotHeight = 0.13
plotWidth = 0.82
plotVSpace = 0.01
plotLeftEdge = 0.10
plotBottomMargin = 0.21
plotHSpace=0.04   #AP Horizontal Space between plots
Widthfactor= .3   #AP for Smaller width plots

histWidth = 0.25*plotWidth
histLeftEdge = plotLeftEdge+plotWidth-histWidth

#------------------------------------------------------------------------------
periodJitterAxes = axes([plotLeftEdge,(4*(plotHeight+plotVSpace)+plotBottomMargin), \
                         plotWidth,plotHeight])
title(fileNameShort, size=14, weight='bold',family = 'monospace')
#scatter(lpr1.edgeTime[minXIndex:maxXIndex]/us, lpr1.pj[minXIndex:maxXIndex]/ps, s=0.1, color='b')
plot(lpr1.edgeTime[minXIndex:maxXIndex]/us, lpr1.pj[minXIndex:maxXIndex]/ps, ',', color='b')
axis([rawGraphStartTime/us,rawGraphEndTime/us,lpr1.pj.min()/ps,lpr1.pj.max()/ps])
ylabel('Period Jitter, ps', size=yLabelSize, family = 'monospace')
ticklabels = periodJitterAxes.get_xticklabels()
ticklabels.extend(periodJitterAxes.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(labelTextSize)
setp(periodJitterAxes, xticks=[])

insetPeriodHistPlot = axes([histLeftEdge,(4*(plotHeight+plotVSpace)+ \
                                          plotVSpace + plotBottomMargin), \
                            histWidth,plotHeight-2*plotVSpace])
hits, hbins, patches = hist(lpr1.pj,100,normed=0,orientation='vertical')
setp(patches, 'facecolor', '0.5', 'alpha', 0.1)
axisbg = 0.75
setp(insetPeriodHistPlot, xticks=[], yticks=[])
#------------------------------------------------------------------------------

def multiPlot(xData,yData,colorNumber):
   N = len(xData)
   DF = 15
   subN = int(N/DF)
   for i in range(0,DF):
      start = i * subN
      finish = start + subN
      plot(xData[start:finish],yData[start:finish],colorNumber)
   return

def multiLogLog(xData,yData,colorNumber,alphaVal):
   N = len(xData)
   DF = 15
   subN = int(N/DF)
   for i in range(0,DF):
      start = i * subN
      finish = start + subN
      loglog(xData[start:finish],yData[start:finish],colorNumber, alpha = alphaVal)
   return

#------------smoothed frequency plot------------
smoothedFrequencyAxes = axes([plotLeftEdge,(3*(plotHeight+plotVSpace)+plotBottomMargin), \
                              plotWidth,plotHeight])
print(len(lpr1.edgeTime),len(lpr1.smoothedFreq16))
print(minXIndex,maxXIndex)
print(len(lpr1.edgeTime[minXIndex:maxXIndex]),len(lpr1.smoothedFreq16[minXIndex:maxXIndex]))
multiPlot(lpr1.edgeTime[minXIndex:maxXIndex]/us, lpr1.smoothedFreq16[minXIndex:maxXIndex]/MHz,'b')
#multiPlot(lpr1.edgeTime[minXIndex:maxXIndex]/us, filteredFreq[minXIndex:maxXIndex]/MHz,'r')
axis([rawGraphStartTime/us,rawGraphEndTime/us,lpr1.minSmoothedFreq16/MHz,lpr1.maxSmoothedFreq16/MHz])

ylabel('F (mov.avg,N=' + '%3d' % (lpr1.nFreqDisplay) + ')', size=yLabelSize, family = 'monospace')
smoothedFrequencyAxes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ticklabels = smoothedFrequencyAxes.get_xticklabels()
ticklabels.extend(smoothedFrequencyAxes.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(labelTextSize)
setp(smoothedFrequencyAxes, xticks=[])
insetFreqHistPlot = axes([histLeftEdge,(3*(plotHeight+plotVSpace)+ \
                                        plotVSpace + plotBottomMargin), \
                        histWidth,plotHeight-2*plotVSpace])
hits, hbins, patches = hist(lpr1.smoothedFreq16,100,normed=0,orientation='vertical')
setp(patches, 'facecolor', '0.5', 'alpha', 0.1)
axisbg = 0.75
setp(insetFreqHistPlot, xticks=[], yticks=[])
#------------------------------------------------------------------------------

#------------phase jitter plot-------------------
##phaseJitterAxes = axes([plotLeftEdge,(2*(plotHeight+plotVSpace)+plotBottomMargin), \
##                        plotWidth,plotHeight])
##multiPlot(lpr1.edgeTime[minXIndex:maxXIndex]/us, lpr1.phi[minXIndex:maxXIndex]/ns, 'b')
##axis([rawGraphStartTime/us,rawGraphEndTime/us,lpr1.phi.min()/ns,lpr1.phi.max()/ns])
##ylabel('Phase Jitter, ns', size=yLabelSize, family = 'monospace')
##ticklabels = phaseJitterAxes.get_xticklabels()
##ticklabels.extend(phaseJitterAxes.get_yticklabels() )
##for label in ticklabels:
##    label.set_fontsize(labelTextSize)
##setp(phaseJitterAxes, xticks=[])
##
insetPhaseJitterPlot = axes([histLeftEdge,(2*(plotHeight+plotVSpace)+ \
                                           plotVSpace + plotBottomMargin), \
                        histWidth,plotHeight-2*plotVSpace])
loglog(lpr1.testDelay,lpr1.LTJResult, color='0.5')
ticklabels = insetPhaseJitterPlot.get_xticklabels()
ticklabels.extend(insetPhaseJitterPlot.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(4)
#------------------------------------------------------------------------------

#------------eye closure plot---------------------
##eyeClosureAxes = axes([plotLeftEdge,(1*(plotHeight+plotVSpace)+plotBottomMargin), \
##                       2*plotWidth,plotHeight])
##xre = PCIe_1.eyeClosureTrend 
###xre = PCIe_3_COMMON_405.eyeClosureTrend                               #JD 20140508
##multiPlot(lpr1.edgeTime[minXIndex:maxXIndex]/us, xre[minXIndex:maxXIndex]/ps,'b')
##axis([rawGraphStartTime/us,rawGraphEndTime/us,xre.min()/ps,xre.max()/ps])
##xlabel('time, us', size=xLabelSize, family = 'monospace')
##ylabel('PCIe Gen1 Eye Cl., ps', size=yLabelSize, family = 'monospace')
###ylabel('PCIe Gen3 Eye Cl., ps', size=yLabelSize, family = 'monospace') #JD 20140508
##ticklabels = eyeClosureAxes.get_xticklabels()
##ticklabels.extend(eyeClosureAxes.get_yticklabels() )
##for label in ticklabels:
##    label.set_fontsize(labelTextSize)

insetEyeClosurePlot = axes([histLeftEdge,(1*(plotHeight+plotVSpace)+ \
                                          plotVSpace*1.2 + plotBottomMargin), \
                        histWidth,plotHeight-2*plotVSpace])
setp(insetEyeClosurePlot, xticks=[], yticks=[])
#------------------------------------------------------------------------------

plotBottomMargin = 0.12
plotHeight = plotHeight + 0.08
#------------------------------------------------------------------------------
#------------spectral purity plot--------------
lowerIndex = int(1e4/lpr1.Res + 0.5)+1           # start at 10KHz

xD = lpr1.freqGrid[lowerIndex:]/MHz
spectralPurityAxes = axes([plotLeftEdge,plotBottomMargin + \
                           29*plotVSpace,plotWidth*.73,plotHeight]) # AP plot location
yD = lpr1.subND[lowerIndex:]/ps
multiLogLog(xD, yD, 'b', 1.0)

axis([0.01,(0.6*lpr1.fMean)/MHz,1e-3,100])
grid(True)
grid.color = '0.5'
xlabel('frequency, MHz', size=xLabelSize, family = 'monospace')
ylabel('Period Jitter FFT, ps', size=yLabelSize, family = 'monospace')

##xlabel('frequency, MHz', size=xLabelSize)
##ylabel('Period Jitter FFT, ps', size=yLabelSize)

ticklabels = spectralPurityAxes.get_xticklabels()
ticklabels.extend(spectralPurityAxes.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(labelTextSize)

resultSize=6
#------------RMS accumulation plot--------------
#     this is the list of specs for which we want to calculate the mean-squared accumulation of
#     phase jitter for jitter accumulation graph

MSphaseWindow = ((cumsum(phaseWindow ** 2))[-1]/len(phaseWindow))
RMSphaseWindow = MSphaseWindow**0.5
#print(MSphaseWindow)

lowerIndex = int(1e4/lpr1.Res + 0.5)+1           # start at 10KHz
xD = lpr1.freqGrid/MHz
rmsAccumulationAxes = axes([plotLeftEdge ,plotBottomMargin \
                           ,plotWidth*Widthfactor,plotHeight])

legendList = []
maxResult = 0



cumMSspecList = (PCIe_1,\
                 PCIe_2A_COMMON_High, \
                 PCIe_2B_COMMON_High
                 #PCIe_2B_COMMON_High, \
                 #PCIe_Gen3_COMMON_2_2_4_2, \
                 #PCIe_Gen3_COMMON_2_1_5_1
                 )
for spec in cumMSspecList:
   spec.calcCumMSPhaseSpectrum(spec.filteredPhaseSpectrum)
   legendList.append(spec.name)
   normFactor = 2 * (1/MSphaseWindow)
   yD = spec.cumMSPhaseSpectrum * normFactor / (ps**2)
   maxResult = max(maxResult,max(yD))
   semilogx(xD, yD)
axis([0.01,(0.6*lpr1.fMean)/MHz,0,maxResult*1.05])

grid(True)
grid.color = '0.5'

legend(legendList, loc='upper left', borderpad=0.0 ,labelspacing=.1)
leg = gca().get_legend()
ltext  = leg.get_texts()  # all the text.Text instance in the legend
setp(ltext, fontsize=resultSize)    # the legend text fontsize

xlabel('frequency, MHz', size=xLabelSize)
ylabel('Cum. Phase Jitter Pwr. (ps^2)', size=yLabelSize)
ticklabels = rmsAccumulationAxes.get_xticklabels()
ticklabels.extend(rmsAccumulationAxes.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(labelTextSize)
#------------------------------------------------------------------------------

#------------RMS accumulation plot--------------
#     this is the list of specs for which we want to calculate the mean-squared accumulation of
#     phase jitter for jitter accumulation graph

MSphaseWindow = ((cumsum(phaseWindow ** 2))[-1]/len(phaseWindow))
RMSphaseWindow = MSphaseWindow**0.5
#print(MSphaseWindow)

lowerIndex = int(1e4/lpr1.Res + 0.5)+1           # start at 10KHz
xD = lpr1.freqGrid/MHz
rmsAccumulationAxes = axes([plotLeftEdge+ plotWidth*Widthfactor +plotHSpace,plotBottomMargin \
                           ,plotWidth*Widthfactor,plotHeight])
legendList = []
maxResult = 0



cumMSspecList = (PCIe_Gen3_COMMON_2_2_4_2, \
                 PCIe_Gen3_COMMON_2_1_5_1
                 )
for spec in cumMSspecList:
   spec.calcCumMSPhaseSpectrum(spec.filteredPhaseSpectrum)
   legendList.append(spec.name)
   normFactor = 2 * (1/MSphaseWindow)
   yD = spec.cumMSPhaseSpectrum * normFactor / (ps**2)
   maxResult = max(maxResult,max(yD))
   semilogx(xD, yD)
axis([0.01,(0.6*lpr1.fMean)/MHz,0,maxResult*1.05])

grid(True)
grid.color = '0.5'

legend(legendList, loc='upper left', borderpad=0.0 ,labelspacing=.1)
leg = gca().get_legend()
ltext  = leg.get_texts()  # all the text.Text instance in the legend
setp(ltext, fontsize=resultSize)    # the legend text fontsize

xlabel('frequency, MHz', size=xLabelSize)
##ylabel('Cum. Phase Jitter Pwr. (ps^2)', size=yLabelSize)
ticklabels = rmsAccumulationAxes.get_xticklabels()
ticklabels.extend(rmsAccumulationAxes.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(labelTextSize)
#------------------------------------------------------------------------------
#------------RMS accumulation plot--------------
#     this is the list of specs for which we want to calculate the mean-squared accumulation of
#     phase jitter for jitter accumulation graph

MSphaseWindow = ((cumsum(phaseWindow ** 2))[-1]/len(phaseWindow))
RMSphaseWindow = MSphaseWindow**0.5
#print(MSphaseWindow)

lowerIndex = int(1e4/lpr1.Res + 0.5)+1           # start at 10KHz
xD = lpr1.freqGrid/MHz
rmsAccumulationAxes = axes([plotLeftEdge+ 2*(plotWidth*Widthfactor +plotHSpace),plotBottomMargin \
                           ,plotWidth*Widthfactor,plotHeight])

legendList = []
maxResult = 0

cumMSspecList = (PCIe_2_SRIS, \
                 PCIe_3_SRIS
                 )
for spec in cumMSspecList:
   spec.calcCumMSPhaseSpectrum(spec.filteredPhaseSpectrum)
   legendList.append(spec.name)
   normFactor = 2 * (1/MSphaseWindow)
   yD = spec.cumMSPhaseSpectrum * normFactor / (ps**2)
   maxResult = max(maxResult,max(yD))
   semilogx(xD, yD)
axis([0.01,(0.6*lpr1.fMean)/MHz,0,maxResult*1.05])

grid(True)
grid.color = '0.5'

legend(legendList, loc='upper left', borderpad=0.0 ,labelspacing=.1)
leg = gca().get_legend()
ltext  = leg.get_texts()  # all the text.Text instance in the legend
setp(ltext, fontsize=resultSize)    # the legend text fontsize

xlabel('frequency, MHz', size=xLabelSize)
##ylabel('Cum. Phase Jitter Pwr. (ps^2)', size=yLabelSize)
ticklabels = rmsAccumulationAxes.get_xticklabels()
ticklabels.extend(rmsAccumulationAxes.get_yticklabels() )
for label in ticklabels:
    label.set_fontsize(labelTextSize)
#------------------------------------------------------------------------------
resultSize=6
#------------------------------------------------------------------------------
timestamp('.......annotating time trend plots')
axes(insetPeriodHistPlot)
XL = xlim()
YL = ylim()
leftTextEdge = XL[0]+(XL[1]-XL[0])*0.02
text(leftTextEdge,(YL[1]-YL[0])*0.84+YL[0], \
        ('Mean Period  = ' + '%.2f' % (lpr1.tMean/ns) +' ns'), \
        weight='bold',  family = 'monospace', size = resultSize)
text(leftTextEdge,(YL[1]-YL[0])*0.72+YL[0], \
        ('Min Period   = ' + '%.2f' % (lpr1.tMin/ns) +' ns'), \
        weight='bold',  family = 'monospace', size = resultSize)
text(leftTextEdge,(YL[1]-YL[0])*0.60+YL[0], \
        ('Max Period   = ' + '%.2f' % (lpr1.tMax/ns) +' ns'), \
        weight='bold',  family = 'monospace', size = resultSize)
text(leftTextEdge,(YL[1]-YL[0])*0.48+YL[0], \
        ('Pk-Pk Jitter = ' + '%.2f' % (lpr1.p2p/ps) +' ps'), \
        weight='bold',  family = 'monospace', size = resultSize)
text(leftTextEdge,(YL[1]-YL[0])*0.36+YL[0], \
        ('RMS Jitter   = ' + '%.2f' % (lpr1.rms/ps) +' ps'), \
        weight='bold',  family = 'monospace', size = resultSize)
text(leftTextEdge,(YL[1]-YL[0])*0.24+YL[0], \
        ('C-C Jitter   = ' + '%.2f' % (lpr1.c2c/ps) +' ps'), \
        weight='bold',  family = 'monospace', size = resultSize)
text(leftTextEdge,(YL[1]-YL[0])*0.12+YL[0], \
        ('Mean C-C Jitt= ' + '%.2f' % (lpr1.meanc2c/ps) +' ps'), \
        weight='bold',  family = 'monospace', size = resultSize)
rambusText = ''
for i in range(1,7):
    rambusText = rambusText + '%.0f' % (lpr1.RambusList[i-1]/ps) + '/'
text(leftTextEdge,(YL[1]-YL[0])*0.01+YL[0], \
        ('1-6 C-C (ps) = ' + rambusText), \
        weight='bold',  family = 'monospace', size = resultSize)
#------------------------------------------------------------------------------

#----------------------------
axes(insetFreqHistPlot)
XL = xlim()
YL = ylim()
leftTextEdge = XL[0]+(XL[1]-XL[0])*0.02
text(leftTextEdge,(YL[1]-YL[0])*0.87+YL[0], \
        ('F Mean              = ' + '%.2f' % (lpr1.fMean/MHz) +' MHz'), \
        weight='bold',   family = 'monospace', size = resultSize)
text(leftTextEdge,(YL[1]-YL[0])*0.75+YL[0], \
        ('F Min/Max (N=16)    = ' + '%.2f' % (lpr1.minSmoothedFreq16/MHz) + '/' \
         + '%.2f' % (lpr1.maxSmoothedFreq16/MHz) +' MHz'), \
        weight='bold',  family = 'monospace', size = resultSize)
text(leftTextEdge,(YL[1]-YL[0])*0.63+YL[0], \
        ('F Min/Max (2MHzLPF) = ' + '%.2f' % (lpr1.minSmoothedFreq2MHz/MHz) + '/' \
         + '%.2f' % (lpr1.maxSmoothedFreq2MHz/MHz) +' MHz'), \
        weight='bold',  family = 'monospace', size = resultSize)
text(leftTextEdge,(YL[1]-YL[0])*0.51+YL[0], \
        ('F Min/Max (T=1us)   = ' + '%.2f' % (lpr1.minSmoothedFreq1u/MHz) + '/' \
         + '%.2f' % (lpr1.maxSmoothedFreq1u/MHz) +' MHz'), \
        weight='bold',  family = 'monospace', size = resultSize)
text(leftTextEdge,(YL[1]-YL[0])*0.39+YL[0], \
        ('pk-pk dev. (N=16)   = ' + '%.2f' % (lpr1.SSCGdeviationPct16)) + ' MHz', \
        weight='bold',  family = 'monospace', size = resultSize)
if maxima_calc==1:
   text(leftTextEdge,(YL[1]-YL[0])*0.27+YL[0], \
           ('Frequency Mod 1/Tp2p Max (KHz)= ' + '%.4f' % (lpr1.Freq_Mod_Period_max)) + '', \
           weight='bold',  family = 'monospace', size = resultSize)
   text(leftTextEdge,(YL[1]-YL[0])*0.15+YL[0], \
           ('Frequency Mod 1/Tp2p Mean(KHz)= ' + '%.4f' % (lpr1.Freq_Mod_Period_mean)) + '', \
           weight='bold',  family = 'monospace', size = resultSize)
   text(leftTextEdge,(YL[1]-YL[0])*0.04+YL[0], \
           ('Frequency Mod 1/Tp2p Min (KHz)= ' + '%.4f' % (lpr1.Freq_Mod_Period_min)) + '', \
           weight='bold',  family = 'monospace', size = resultSize)
##text(leftTextEdge,(YL[1]-YL[0])*0.25+YL[0], \
##        ('Frequency Minima Max   = ' + '%.6f' % (lpr1.minima_max/MHz)) + ' %', \
##        weight='bold',  family = 'monospace', size = resultSize)


#------------------------------------------------------------------------------

#----------------------------
axes(insetPhaseJitterPlot)
XL = xlim()
YL = ylim()
leftTextEdge = XL[0]+(XL[1]-XL[0])*0.02
text(lpr1.testDelay[1],lpr1.LTJResult[1], \
         ('LTJ(10u) = ' + '%.2f' % (lpr1.LTJ_10u/ps) +'ps'), \
        weight='bold',  family = 'monospace', size = resultSize)
#------------------------------------------------------------------------------

#----------------------------
axes(insetEyeClosurePlot)
XL = xlim()
YL = ylim()
leftTextEdge = XL[0]+(XL[1]-XL[0])*0.02
text(leftTextEdge,(YL[1]-YL[0])*0.85+YL[0], \
        ('Eye Closures in ps '), family = 'monospace', weight='bold', size = resultSize)
text(leftTextEdge,(YL[1]-YL[0])*0.70+YL[0], \
        ('PCIe 1 "revised" (p-p/RMS)  = ' + \
         '%.2f' % (PCIe_1.p2p/ps) + '/' + \
         '%.2f' % (PCIe_1.rms/ps)), family = 'monospace', weight='bold', size = resultSize)
text(leftTextEdge,(YL[1]-YL[0])*0.55+YL[0], \
        ('PCIe 2 (lo/hi band) RMS     = ' +  \
         '%.2f' % (PCIe_2_WORST_rms_loBand/ps) + '/' + \
         '%.2f' % (PCIe_2_WORST_rms_hiBand/ps)), family = 'monospace', weight='bold', size = resultSize)
text(leftTextEdge,(YL[1]-YL[0])*0.4+YL[0], \
        ('PCIe 2 SRIS  RMS            = '   + '%.2f' % (PCIe_2_SRIS_WORST_rms/ps) ), family = 'monospace', weight='bold', size = resultSize)
text(leftTextEdge,(YL[1]-YL[0])*0.25+YL[0], \
        ('PCIe 3 RMS                  = '   + '%.2f' % (Gen3_PCIe_3_WORST_rms/ps) ), family = 'monospace', weight='bold', size = resultSize)
text(leftTextEdge,(YL[1]-YL[0])*0.1+YL[0], \
        ('PCIe 3 SRIS  RMS            = '   + '%.2f' % (PCIe_3_SRIS_WORST_rms/ps) ), family = 'monospace', weight='bold', size = resultSize)

#------------------------------------------------------------------------------

#----------------------------
axes(spectralPurityAxes)
XL = xlim()
YL = ylim()
text(0.12,30, \
        ('JIT3 compatible - see .csv file for details'), \
        weight='bold', size = 6)

topSpur_MHz_ps = []
minFreqForSpurPlot = 1.0*MHz
minMagForSpurPlot = 1.0*ps

for i in range(0,lpr1.numPeaks):
    if (lpr1.AssocFr[i] > minFreqForSpurPlot) and (lpr1.MaxAmpl[i] > minMagForSpurPlot):
        plotIt = True
        for j in range(0,i):
            if (lpr1.AssocFr[j] > 0.98*lpr1.AssocFr[i]) and \
               (lpr1.AssocFr[j] < 1.02*lpr1.AssocFr[i]) and \
               (lpr1.MaxAmpl[i] > 0.5*lpr1.MaxAmpl[j]):
                plotIt = False
        if plotIt:
            plotPoint = (lpr1.AssocFr[i]/MHz,lpr1.MaxAmpl[i]/ps)
            topSpur_MHz_ps.append(plotPoint)

for i in range(0,len(topSpur_MHz_ps)):
    thisPoint = topSpur_MHz_ps[i]
    text(thisPoint[0],thisPoint[1]*2, '%3.2f' % thisPoint[1], size = 5)
#----------------------------

figtext(0.1,0.01,'Plot generated on ' + time.asctime() + ' by: ' + \
        thisProgram + ': Rev ' + revLevel + ', Date: ' + revDate + ' -- IDT CONFIDENTIAL INFORMATION', size = 7)

figtext(0.1,0.03,('directory: ' + pathName), size = 7)


savefig(plot1FileName, dpi=150)
show()
close()

###------------Spectrum Plot--------------
if spectrum_plot:

   plotHeight = .8 
   plotWidth = .8
   plotVSpace = 0.01
   plotLeftEdge = 0.1
   plotBottomMargin = 0.1

   histWidth = 0.25*plotWidth
   histLeftEdge = plotLeftEdge+plotWidth-histWidth

   lowerIndex = int(1e4/lpr1.Res + 0.5)+1           # start at 10KHz

   xD = lpr1.freqGrid[lowerIndex:]/MHz
   spectralPurityAxes = axes([plotLeftEdge,plotBottomMargin \
                              ,plotWidth,plotHeight])
   ##yD = lpr1.subND[lowerIndex:]/ps
   sizeN= len(PCIe_Gen3_COMMON_2_2_4_2.BPFfilteredPhaseSpectrum)
   absspectrum = abs(PCIe_3_SRIS.filteredPhaseSpectrum) *2  #/1e12*1e0
   ##absspectrum = abs(PCIe_3_SRIS.eyeClosureTrend) *2 /100
   
   ##absspectrum = abs(PCIe_3_SRIS.BPFfilteredPhaseSpectrum)/1e12*1e3 

   absspectrumcut = absspectrum[0:N/2+1]
   yD = absspectrumcut[lowerIndex:]/ps
   multiLogLog(xD, yD, 'y', 1.0)

   axis([0.01,(0.6*lpr1.fMean)/MHz,1e-3,100])
   grid(True , which="both")
   grid.color = '0.5'
   xlabel('frequency, MHz', size=xLabelSize, family = 'monospace')
   ylabel('Period Jitter FFT, ps ', size=yLabelSize, family = 'monospace')
   ##ylabel('Filter Response Gen 1 ComClk', size=yLabelSize, family = 'monospace')
   ticklabels = spectralPurityAxes.get_xticklabels()
   ticklabels.extend(spectralPurityAxes.get_yticklabels() )
   for label in ticklabels:
       label.set_fontsize(labelTextSize)

   savefig(plot2FileName, dpi=150)
   show()
   close()
   
#------------eye closure plot---------------------

if Eye_Closure_plot:
   
   plotHeight = .8
   plotWidth = .7
   plotVSpace = 0.01
   plotLeftEdge = 0.10
   plotBottomMargin = 0.11
   plotHSpace=0.04   #AP Horizontal Space between plots

   eyeClosureAxes = axes([plotLeftEdge,plotBottomMargin, \
                          plotWidth,plotHeight])
   xre = PCIe_3_SRIS.eyeClosureTrend 
##   xre = lpr1.phi
   #xre = PCIe_2A_COMMON_Low.eyeClosureTrend                               #JD 20140508
##   multiPlot(lpr1.edgeTime[minXIndex:maxXIndex]/us, xre/ps,'b')
   multiPlot(lpr1.edgeTime[minXIndex:maxXIndex]/us, xre/ps,'b')
   ##multiPlot(lpr1.edgeTime[minXIndex:maxXIndex]/us, xre[minXIndex:maxXIndex]/ps,'b')
   axis([rawGraphStartTime/us,rawGraphEndTime/us,xre.min()/ps,xre.max()/ps])
   xlabel('time, us', size=xLabelSize, family = 'monospace')
   ylabel('PCIe Gen1 Eye Cl., ps', size=yLabelSize, family = 'monospace')
   #ylabel('PCIe Gen3 Eye Cl., ps', size=yLabelSize, family = 'monospace') #JD 20140508
   ticklabels = eyeClosureAxes.get_xticklabels()
   ticklabels.extend(eyeClosureAxes.get_yticklabels() )
   for label in ticklabels:
       label.set_fontsize(labelTextSize)

   savefig(plot2FileName, dpi=150)
   show()
   close()
   
timestamp('....done')




