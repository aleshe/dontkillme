import numpy as np
import matplotlib.pyplot as plt
import sparse_stoch_mat_central as sp_stoch_mat
import sys
from scipy.sparse.linalg import spsolve
import timeit
#sys.path.append('/home/hcleroy/Extra_Module_py/')
#from formated_matplotlib import *
from functions import *


class physfcts:

    def __init__(self, array):
        self.eta, self.kab0, self.kbc0, self.kac0, self.Eab0, self.Ebc0, self.Eac0, self.V0a, self.V0b, self.V0c, self.V1a, self.V1b, self.V1c, self.Aab, self.Abc, self.Aac, self.signA, self.signB, self.signC, self.per, self.X0, self.Xf, self.n, self.dx = array
    
    def Eab(self,x):
        return self.Eab0 + self.eta* (self.V1a*np.exp(self.V0a)+self.V1b*np.exp(self.V0b)) * (np.sin(self.per*np.pi*x/(2*self.Xf)))
    
    def Ebc(self, x): 
        return self.Ebc0 + self.eta* (self.V1b*np.exp(self.V0b)+self.V1c*np.exp(self.V0c)) * (np.sin(self.per*np.pi*x/(2*self.Xf)))
    
    def Eac(self, x): 
        return self.Eac0 + self.eta* (self.V1a*np.exp(self.V0a)+self.V1c*np.exp(self.V0c)) * (np.sin(self.per*np.pi*x/(2*self.Xf)))
    # define all the Vs :
    
    def Va_out(self, x) : 
        return self.V0a + self.eta* self.signA*(np.sin(self.per*np.pi*x/(2*self.Xf))) * self.V1a
    
    def Vb_out(self, x) : 
        return self.V0b + self.eta* self.signB*(np.sin(self.per*np.pi*x/(2*self.Xf))) * self.V1b
    
    def Vc_out(self, x) : 
        return self.V0c + self.eta* self.signC*(np.sin(self.per*np.pi*x/(2*self.Xf))) * self.V1c
    # define all the k's
    
    def kab_out(self, X, *arg): 
        if type(X) == np.ndarray:
            out = np.array([self.kab0 * np.exp(-self.Eab(x)+self.Va_out(x))*arg[0] for x in X])
        else:
            out = self.kab0 * np.exp(-self.Eab(X)+self.Va_out(X))*arg[0] # arg is the activity
        return out
    
    def kba_out(self, X, *arg) : 
        if type(X)==np.ndarray:
            out = np.array([self.kab0 * np.exp(-self.Eab(x)+self.Vb_out(x))for x in X])
        else:
            out = self.kab0 * np.exp(-self.Eab(X)+self.Vb_out(X))# no activity but it takes the same argument as kab
        return out
    
    def kbc_out(self, X, *arg) : 
        if type(X)==np.ndarray:
            out = np.array([self.kbc0 * np.exp(-self.Ebc(x)+self.Vb_out(x))*arg[0]for x in X]) 
        else: 
            out = self.kbc0 * np.exp(-self.Ebc(X)+self.Vb_out(X))*arg[0]# arg is the activity
        return out
    
    def kcb_out(self, X, *arg) : 
        if type(X) == np.ndarray:
            out = np.array([self.kbc0 * np.exp(-self.Ebc(x)+self.Vc_out(x))for x in X])  
        else: 
            out = self.kbc0 * np.exp(-self.Ebc(X)+self.Vc_out(X))# no activity but it takes the same argument as kbc
        return out
    
    def kac_out(self, X, *arg) : 
        if type(X) ==np.ndarray:
            out = np.array([self.kac0 * np.exp(-self.Eac(x)+self.Va_out(x))*arg[0]for x in X])
        else: 
            out = self.kac0 * np.exp(-self.Eac(X)+self.Va_out(X))*arg[0]# arg is the activity
        return out
    
    def kca_out(self, X, *arg) : 
        if type(X) == np.ndarray:
            out = np.array([self.kac0 * np.exp(-self.Eac(x)+self.Vc_out(x))for x in X])  
        else:
            out = self.kac0 * np.exp(-self.Eac(X)+self.Vc_out(X))# no activity but it takes the same argument as kac
        return out
    
    def NumRhos(self):
        OoeStochMat = sp_stoch_mat.make_transition_matrix(self.Va_out,self.Vb_out,self.Vc_out,
                                        self.kab_out,self.kba_out,self.kbc_out,self.kcb_out,self.kac_out,self.kca_out,
                                        self.kab0,self.kbc0,self.kac0,
                                        self.Aab,self.Abc,self.Aac,
                                        X0=self.X0,Xf=self.Xf,n=int(self.n))
        rhoaooe,rhobooe,rhocooe = get_kernel_stoch_mat(OoeStochMat,self.dx)
        return rhoaooe, rhobooe, rhocooe
    
    def MathRhos(self, x):
        expAB = np.exp(self.Eab0)
        expAC = np.exp(self.Eac0)
        expBC = np.exp(self.Ebc0)
        pi = np.pi
        eta, Aab, Abc, Aac = self.eta, self.Aab, self.Abc, self.Aac
        V1a, V1b, V1c = self.V1a, self.V1b, self.V1c
        signA, signB, signC, per = self.signA, self.signB, self.signC, self.per

        overpref = 2*((1+2*Aac)*expAB + 3*expAC + (2+Aac)*expBC)
        #checked, correct
        firstpartA = expAB + expBC + expAC
        firstpartB = Aac*expAB + expBC + expAC
        firstpartC = Aac*expAB + Aac*expBC + expAC
        
        sinmultiplierA = eta*(-16 *(-1 + Aac)* expAC* expBC* V1a + 
        expAB**2* (16 + 32* Aac - 4 *(2* expAC + expBC + Aac* expBC)* per**2* pi**2 + 
        expAC* expBC* per**4* pi**4)* signA* V1a - 
        8* (expAC + expBC)* (-2 *(2 + Aac)* expBC + 
        expAC* (-6 + expBC* per**2* pi**2))* signA* V1a + 
        16 *(-1 + Aac)* expAC* expBC* V1b + 
        expAB* (8* expAC* (8 + 4* Aac - expAC* per**2* pi**2)* signA* V1a + 
        expBC**2* per**2* pi**2* (-4 - 4* Aac + expAC* per**2* pi**2)* signA* V1a + 
        expBC* (-16 + (48 + expAC* per**2* pi**2* (-20 + expAC *per**2* pi**2))* signA + 
        Aac* (16 + 48* signA - 4* expAC *per**2* pi**2* signA))* V1a + 
        32 *(-1 + Aac)* expAC* (V1b - V1c) - 
        4 *(-1 + Aac)* expBC* (expAC* per**2* pi**2* (V1b - V1c) + 4* V1c)))
        #checked, correct

        sinmultiplierB = eta*(-16 *(-1 + Aac)* expAC* expBC* V1a + 16* (-1 + Aac)* expAC* expBC *V1b + 
        Aac* expAB**2* (16 + 32 *Aac - 4* (2 *expAC + expBC + Aac* expBC)* per**2* pi**2 + 
        expAC* expBC* per**4* pi**4)* signB* V1b - 8* (expAC + expBC)* (-2* (2 + Aac)* expBC + 
        expAC* (-6 + expBC* per**2* pi**2))* signB* V1b + 
        expAB* (expBC**2* per**2* pi**2* (-4 - 4* Aac + expAC* per**2* pi**2)* signB* V1b + 
        8* expAC *((2 - 2* Aac + 2* signB + 10* Aac* signB - 
        expAC* per**2* pi**2* signB)* V1b + 2* (-1 + Aac)* V1c) + 
        expBC* (-4 *(-1 + Aac)* (4 + 4 *Aac - expAC* per**2* pi**2)* V1a + (16 + 
        16* Aac* (4 + Aac) - 12* (1 + Aac)* expAC *per**2* pi**2 + 
        expAC**2* per**4* pi**4)* signB* V1b + 
        4* (-1 + Aac)* (4 + 4* Aac - expAC* per**2* pi**2)* V1c)))
        #checked, correct
        sinmultiplierC = -eta*(-expAC* (16 *expAB* V1b + 48 *expAC* signC* V1c - 
        8* expAB* (2 + (-2 + expAC* per**2* pi**2)* signC)* V1c + 
        expBC* (-8 + expAB* per**2* pi**2)* (4 *V1a - 4* V1b + (-4 + expAC* per**2* pi**2)* signC* V1c)) + 
        4* Aac**2* (-4* expBC**2* signC* V1c + expAB**2* (-8 + expBC* per**2* pi**2)* signC* V1c + 
        expAB* expBC* (-4* V1a + (4 - 12* signC + expBC* per**2* pi**2* signC) *V1c)) - 
        Aac* (expAB**2* (16 - 4 *expBC* per**2* pi**2 + expAC* per**2* pi**2* (-8 + expBC* per**2* pi**2)) *signC* V1c + 
        expAB* (-4 *expBC* (4 + expAC* per**2* pi**2)* V1a + 
        expBC**2* per**2* pi**2* (-4 + expAC* per**2* pi**2)* signC* V1c + 
        4* expAC* expBC* per**2* pi**2* (V1b - 5* signC* V1c) + 
        16* expBC* (V1c + 3* signC* V1c) + 16* expAC* (-V1b + V1c + 5* signC* V1c)) - 
        8* expBC* (-4* expBC* signC* V1c + expAC* (-4* V1a + 4 *V1b + (-8 + expBC* per**2* pi**2)* signC *V1c))))

        denum = (-16* (expAB + 2* Aac* expAB + 3 *expAC + (2 + Aac)* expBC) + 
        4* (2 *expAC* expBC + expAB* (2* expAC + expBC + Aac* expBC))* per**2* pi**2 -
        expAB* expAC* expBC* per**4* pi**4)
        rho_a_m = lambda X: np.array([1/overpref*(firstpartA + np.sin(self.per *np.pi* x/2)*sinmultiplierA/denum) for x in X])
        rho_b_m = lambda X: np.array([1/overpref*(firstpartB + np.sin(self.per *np.pi* x/2)*sinmultiplierB/denum) for x in X])
        rho_c_m = lambda X: np.array([1/overpref*(firstpartC + np.sin(self.per *np.pi* x/2)*sinmultiplierC/denum) for x in X])
        return rho_a_m(x), rho_b_m(x), rho_c_m(x)

    def diff_flux(self,rho,V):
        """"
        - rho : is the vector of density in space
        - V is a vector with the value of the potential in the space
        - rho[i] and V[i] are assumed to correspond to the same point in space
        - derivatives at the boundaries are assumed to be 0
        """
        dVx = D(V,self.dx)
        return D(rho,self.dx)+rho*dVx

    def flux_cc(self,x):
        rhoa, rhob, rhoc = self.NumRhos()
        flux = self.kba_out(x)*rhob+self.kac_out(x,self.Aac)*rhoa+self.kcb_out(x)*rhoc
        return flux

    def flux_cw(self,x):
        rhoa, rhob, rhoc = self.NumRhos()
        flux = self.kbc_out(x,self.Abc)*rhob+self.kca_out(x)*rhoc+self.kab_out(x,self.Aab)*rhoa
        return flux
    
    def chem_flux(self, x):
        rhoA, rhoB, rhoC = self.NumRhos()
        kAB, kBA = self.kab_out(x, self.Aab), self.kba_out(x)
        kBC, kCB = self.kbc_out(x, self.Abc), self.kcb_out(x)
        kAC, kCA = self.kac_out(x, self.Aac), self.kca_out(x)
        fluxA = -rhoA*(kAB+kAC) + rhoB*kBA + rhoC*kCA
        fluxB = -rhoB*(kBA+kBC) + rhoA*kAB + rhoC*kCB
        fluxC = -rhoC*(kCA+kCB) + rhoB*kBC + rhoA*kAC
        return fluxA, fluxB, fluxC

    def flux_ana(self,x):
        expAB = np.exp(self.Eab0)
        expAC = np.exp(self.Eac0)
        expBC = np.exp(self.Ebc0)
        pi = np.pi
        eta, Aab, Abc, Aac = self.eta, self.Aab, self.Abc, self.Aac
        V1a, V1b, V1c = self.V1a, self.V1b, self.V1c
        signA, signB, signC, per = self.signA, self.signB, self.signC, self.per

        denum = (expAB + 2 *Aac* expAB + 3* expAC + (2 + Aac)* expBC)* (-16* (expAB + 2 *Aac* expAB + 
        3* expAC + (2 + Aac)* expBC) + 4* (2* expAC* expBC + 
        expAB* (2 *expAC + expBC + Aac* expBC)) *per**2* pi**2 - expAB *expAC* expBC* per**4* pi**4)

        pref = (-1 + Aac) *eta *pi *per

        Apart = 4* expAC* expBC* (-V1a + V1b) + 8* expAB* expAC* (V1b - V1c) + expAB* expBC *(4* V1a - 4 *V1c + expAC* per**2 *pi**2 *(-V1b + V1c))
        Bpart = 4* expAC *expBC* (V1a - V1b) + expAB *expBC* (4 + 4* Aac - expAC* per**2* pi**2)* (V1a - V1c) + 4 *expAB *expAC* (V1b - V1c)
        Cpart = -expAC *expBC* (-8 + expAB* per**2* pi**2)* (V1a - V1b) + 4* Aac* expAB* expBC* (V1a - V1c) + 4* expAB* expAC* (-V1b + V1c)

        fluxA = pref/denum*Apart*np.cos(per*pi*x/2) 
        fluxB = -pref/denum*Bpart*np.cos(per*pi*x/2) 
        fluxC = pref/denum*Cpart*np.cos(per*pi*x/2) 
        return fluxA, fluxB, fluxC