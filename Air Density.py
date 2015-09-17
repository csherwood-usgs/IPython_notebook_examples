
# coding: utf-8

# <h2>Calculate air density from temperature, pressure, and relative humidity</h2>
# http://wahiduddin.net/calc/density_altitude.htm

# In[3]:

def pvs(T):
    # Calculate saturation water vapor pressure (hPa==millibar)
    # T = air temperature at dewpoint (degrees Celsius)
    # (per Herman Wobus according to several web sources)
    eso = 6.1078
    c0 = 0.99999683
    c1 = -0.90826951e-2
    c2 = 0.78736169e-4
    c3 = -0.61117958e-6
    c4 = 0.43884187e-8
    c5 = -0.29883885e-10
    c6 = 0.21874425e-12
    c7 = -0.17892321e-14
    c8 = 0.11112018e-16
    c9 = -0.30994571e-19
    p = c0 +T*(c1+T*(c2+T*(c3+T*(c4+T*(c5+T*(c6+T*(c7+T*(c8+T*(c9)))))))))
    return eso/(p**8)
def pv(T,p,RH):
    # Actual water vapor pressure for T (deg C), p (millibars) at relative humidity RH (%)
    return pvs(T)*(RH/100.)
def rho(T,p,RH):
    # Air density at temperature T (deg C), pressure p (millibars), and relative humidty RH (%)
    return (p*100.)/ (287.05*(T+273.15)) *(1.-(0.378*pv(T,p,RH))/(p*100.))
T = 20.
p = 1013.25
RH = 40.
print pvs(T)
print pv(T,p,RH)
print "Basement w/ dehumidifier: ",rho(17,p,40.)," kg/m3"
print "Upstairs when muggy:",rho(27.,p,80.)," kg/m3"
print "Dry air is denser than moist air!"

