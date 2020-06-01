import numpy as np
import matplotlib.pyplot as plt
import numba
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import minimize
from lbl_ir.GPR import GPR_engine

"""
Here I try to use Gaussian process regression for identifying peaks
and estimate their standard deviation. 
I build a simple GPR fitter myself because I needed more control over 
kernel and hyper parameters. Things need to be cleaned up.

"""


#@numba.jit(nopython=True)
def dmat(X1,X2=None):
    if X2 is None:
        X2 = X1
    N1,M1 = X1.shape
    N2,M2 = X2.shape 

    dd = np.zeros( (N2,N1) )
    for ii in range(N2):
        for jj in range(N1):
            tmp = 0
            for kk in range(M1):
                delta = X2[ii,kk]-X1[jj,kk]
                tmp += delta*delta
            tmp = np.sqrt( tmp )
            dd[ii,jj]=tmp
    return dd

@numba.jit(nopython=True)
def rbf_kern(dmat, variance, length):
    result = np.exp( -dmat*dmat/(length*length*2.0))*variance
    return result

def d_rbf_kerni_FD(X2, X, variance, length, h=1e-4):
    dmat_plus = dmat(X,X2+h/2)
    dmat_minus= dmat(X,X2-h/2)
    Kplus     = np.exp( -dmat_plus*dmat_plus/(length*length*2.0) )*variance
    Kminus    = np.exp( -dmat_minus*dmat_minus/(length*length*2.0) )*variance
    result    = (Kplus - Kminus)/h
    return result

def d_rbf_kern(X2, X, dxx2, Kxx2, variance, length):
    # first derivative
    result1 = []
    for xx in X2.flatten():
        result1.append( X.flatten()-xx )
    result1 = np.vstack(result1)
    result1 = result1*Kxx2

    # second derivative
    result2 = []
    N2,N1 = result1.shape
    for ii in range(N2):
        xx = X2[ii,:]
        dK = result1[ii,:]
        K  = Kxx2[ii,:]
        tmp = -K/(length*length)  
        tmp2 = (X-xx).flatten()*dK.flatten()/(length*length)
        result2.append( tmp.flatten()+tmp2.flatten() )
    result2 = np.vstack(result2)
    return result1, result2

 

class peak_picker(object):
    def __init__(self,X,Y,sY):
        self.X = X
        self.Y = Y
        self.N = X.shape[0]
        self.sY = sY
        init_scale = np.std( self.X.flatten())
        obj = GPR_engine.GPR_exp_fitter( sigma=np.std(Y.flatten()), length=init_scale)
        obj.fit( self.X, self.Y, self.sY )

        self.lengthscale = obj.length
        self.variance    = obj.sigma**2.0
        self.mu          = obj.mu

        # set things up for peak picking purposes
        self.dXX      = dmat(self.X)
        self.Kxx      = rbf_kern(self.dXX,self.variance,self.lengthscale)
        self.Keff     = np.eye(  self.N )*self.sY*self.sY + self.Kxx
        self.Keff_inv = np.linalg.pinv(self.Keff,rcond=1e-10)
        self.Keff_invY= self.Keff_inv.dot(self.Y-self.mu)

        self.dx = np.mean( np.sort( self.dXX.flatten() )[self.N:self.N*2] )




    def predict(self,x, grads=True):
        # setup stuff
        dxx2     = dmat(self.X, x )
        dx2x2    = dmat(x) 
        Kxx2     = rbf_kern( dxx2, self.variance,self.lengthscale )
        Kx2x2    = rbf_kern( dx2x2, self.variance, self.lengthscale )

        # mean and variance
        cond_mean     = Kxx2.dot( self.Keff_invY ) + self.mu
        cond_variance = Kx2x2 - Kxx2.dot(self.Keff_inv).dot(Kxx2.transpose())

        if grads:
            # derivatives if requested
            dK,ddK = d_rbf_kern(x, self.X, dxx2, Kxx2, self.variance, self.lengthscale)
            dmu = dK.dot( self.Keff_invY )
            ddmu = ddK.dot( self.Keff_invY )
            return cond_mean, cond_variance, dmu, ddmu

        else:
            return cond_mean, cond_variance

    def f(self,x):
        xx = np.array([x]).reshape(-1,1)
        result = self.predict(xx)[0][0,0]+self.mu
        return -result


    def find_peak(self, x, eps=1e-2, max_iter=1000):
        init_simplex = np.array( [x, x+eps] ).reshape(-1,1)
        x_opt = minimize(fun=self.f, method='Nelder-Mead',x0=x, options={'initial_simplex':init_simplex} )
        return x_opt.x

    def find_peak_oof(self, x, eps=1e-8,max_iter=1000, damp=1.0, peak_range=None):
        # do a simple root finding starting from x
        converged=False
        x_in = np.array(x).reshape(-1,1)
        iter = 0
        restart = 0
        max_delta = 1.0
        if peak_range is not None:
            max_delta = np.max_peak(range) - np.min(peak_range)
            max_delta = max_delta / 10.0
        while not converged:
            m,v,dm,ddm = self.predict(x_in,True)
            tmp = np.random.uniform(0.95,1.05,1)[0]
            #print("HERE",m,v,dm,ddm)
            delta = tmp*dm/ddm
            #if np.abs(delta) > max_delta:
            #    delta = max_delta*(np.sign(delta))
            #print(x_in, delta, dm, ddm,"DELTA", x_in - delta*damp )
            x_in = x_in - delta*damp
            iter+=1
            if np.abs(dm) < eps:
                converged=True
            if iter > max_iter:
                converged = False
                print("Early termination in peak finding. Apply damping")
                restart += 1
                iter  = 0
                x_in = np.array(x).reshape(-1,1)
                damp = damp * 0.5
            if restart > 5:
                print("Early termination in peak finding. Damping doesn't work")
                converged = True
                return None
        return x_in[0,0]


    def fst_der_prop(self,x):
        x0 = np.array([x]).reshape(-1,1)
        # we need to get the vector 
        M = self.X.flatten()-x
        M = np.diag(M/self.lengthscale*self.lengthscale)
              
        dxx2     = dmat(self.X, x0 )
        dx2x2    = dmat(x0)
        Kxx2     = rbf_kern( dxx2, self.variance,self.lengthscale )
        Kx2x2    = rbf_kern( dx2x2, self.variance, self.lengthscale )
        nmu      = M.dot(Kxx2.transpose()).transpose().dot(self.Keff_invY) 
        tmp      = M.dot(Kxx2.transpose()).transpose()
        nmu      = tmp.dot(self.Keff_invY)    
        nvar     = Kx2x2 - tmp.dot(self.Keff_inv).dot(tmp.transpose())
        return nmu[0][0], nvar[0][0]

    def peak_and_std_via_resample(self,x_start,N=10, factor = 0.75):
        x_peak = self.find_peak( x_start )
        h = self.dx*factor
        x_around = np.array( [x_peak-h, x_peak, x_peak+h] ).reshape(-1,1)
        # collect the new posterior mean and variance
        mu, var = self.predict(x_around, grads=False) 
        fs = np.random.multivariate_normal(  mu.flatten(), var, N )
        ds = []
        for ii in range(N):
            c = Polynomial.fit( x_around.flatten()-x_peak, fs[ii,:], deg = 2 )   
            c = c.coef        
            dx = -0.5*c[1]/c[2]
            ds.append(dx)
        return x_peak, np.std( ds ), mu[1], np.sqrt(np.abs(var[1][1]))


def tst(P=8,S=0.1, show=False):
    X = np.random.uniform(-3,3,P).reshape(-1,1) # np.linspace(-3,3,P).reshape((P,1))
    X = np.linspace(-3,3,P).reshape((P,1))
    Y = 10.0*np.exp(-X*X) + np.random.normal(0,S, (P,1) )
 
    this_one = np.argmax( Y.flatten() )
    this_x = X.flatten()[this_one]
    S = np.zeros( P )+S
    obj = peak_picker(X,Y,S)
    peak,sigma,val,sig = obj.peak_and_std_via_resample(this_x, N=100)
    assert abs(peak/sigma) <  4
    print( "OK" ) 
    if show:
        Xstar = np.linspace(-3,3,1024).reshape(-1,1)
        y,v,dm,ddm = obj.predict(Xstar, True)
        plt.plot( X.flatten()    , Y.flatten(), '.' )
        plt.plot( Xstar.flatten(), y.flatten(), '-' )
        #plt.plot( Xstar.flatten(), dm.flatten(), '--', lw=3)
        #plt.plot( Xstar.flatten(), ddm.flatten(), '--', lw=2)
        print(peak) 
        peak_val, peak_std = obj.predict(np.array([[peak[0]]]), False)
        print(peak_val, peak_std)
        plt.errorbar( peak, peak_val, peak_std, sigma )# '.', markersize=10)
        plt.show()

 

 

if __name__ =="__main__":
    tst(P=35,S=0.25,show=True) 
