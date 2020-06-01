import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.optimize import minimize
from scipy.optimize import basinhopping

@numba.jit(nopython=True) #,cache=True)
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

@numba.jit(nopython=True) #,cache=True)
def rbf_kern(dmat, variance, length):
    result = np.exp( -dmat*dmat/(1e-8+length*length*2.0))*(1e-8+variance)
    return result

@numba.jit(nopython=True) #,cache=True)
def d_rbf_kern_FD(X2, X, variance, length, h=1e-4):
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



class GPR_exp_fitter(object):
    def __init__(self,sigma,length, mu=0,niter=3):
        """
        :param sigma: The sigma
        :param length: The length scale
        :param mu: The mean value
        :param niter: the number of iterations in basin hopping to fit the GPR model
        """

        self.sigma = sigma
        self.length = length
        self.mu = mu

        # to compute later
        self.dX1X1 = None


    def marginal_likelihood(self,hparams):
        this_mu    = hparams[0]
        this_sigma = hparams[1]
        this_length= hparams[2]
        KX1X1      = rbf_kern( self.dX1X1, this_sigma*this_sigma, this_length ) 

        II         = np.diag(self.sY*self.sY)
        KX1X1_inv  = np.linalg.pinv(KX1X1+II)
        tY         = self.Y - this_mu
        tY         = tY.reshape(-1,1)
        KiY        = KX1X1_inv.dot(tY)
        term1      = 0.5*tY.transpose().dot(KiY)    
        term2      = 0.5*np.linalg.slogdet(KX1X1+II)[1]

        result     = term1+term2
        result     = result.flatten()[0]
        return result


    def fit(self,X,Y,sY):
        # first we need to build a distance matrix
        self.sY = sY
        if type(self.sY) is float:
            self.sY = self.Y*0+sY
        self.X = X
        self.Y = Y
        self.dX1X1 = dmat(X.reshape(-1,1))
        fitter = basinhopping( func   = self.marginal_likelihood,
                               x0     = np.array([self.mu, self.sigma, self.length]),
                               niter  = 13)
        self.mu      = fitter.x[0]
        self.sigma   = abs(fitter.x[1])
        self.length  = abs(fitter.x[2])


        self.tY    = (Y - self.mu).reshape(-1,1)
        self.KX1X1 = rbf_kern( self.dX1X1, self.sigma**2.0, self.length ) + np.eye( self.dX1X1.shape[0])*self.sY
        self.KX1X1_inv = np.linalg.pinv(self.KX1X1)
        self.KiY = self.KX1X1_inv.dot(self.tY)
          


    def predict(self,X):
        dX1X2 = dmat(self.X.reshape(-1,1), X.reshape(-1,1))
        kX1X2 = rbf_kern( dX1X2, self.sigma**2, self.length )
        tmp   = kX1X2.dot( self.KiY )
        return tmp+self.mu  
        
  


def tst():
    x   = np.linspace(-10,10,20, True)
    xx  = np.linspace(-10,10,200,True)
    y   = np.sin(x) #10 + x*x + x

    plt.plot(x,y,'.');plt.show()
    obj = GPR_exp_fitter(1,1,1) 
    obj.fit(x,y,0.01)
    yy = obj.predict( xx ).flatten()
    plt.plot(x,y,'.'); plt.plot(xx,yy);plt.show()


if __name__ == "__main__":
    tst()
