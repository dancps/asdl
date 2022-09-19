import numpy as np; np.set_printoptions(precision=3)
import numpy.random as rnd
import matplotlib.pyplot as plt
from tabulate import tabulate
import os

class DynamicSystem:
    def __init__(self,A,inital_contidition,verbose=False,debug=False):
        self.A = A
        self.__verbose = verbose
        self.__debug = debug
        self.det = np.linalg.det(self.A) 
        self.trace = np.trace(A)

        self.characteristic = self.get_characteristic()
        self.eigenval = self.get_eigenval(verbose = self.__verbose, debug = self.__debug )
        self.eigenvec = self.get_eigenvec(verbose = self.__verbose, debug = self.__debug )
        
        self.inital_contidition = inital_contidition
        self.C = self.get_system()
        self.X_dot = self.A@self.inital_contidition
        self.pointcare_class = self.get_pointcare_classification()
        self.stability = self.get_stability()
        self.pointcare_class = self.get_pointcare_classification()
        
    def set_verbosity(self,verbose):
        if(isinstance(verbose,bool)):
            self.__verbose
        else:
            raise ValueError(f"The value must be boolean.")
    def set_debug(self,debug):
        if(isinstance(debug,bool)):
            self.__debug
        else:
            raise ValueError(f"The value must be boolean.")
    def get_characteristic(self, matrix=None):
        """[summary]

        Args:
            A (np.array[(n,n)]): Matriz de estados
            verbose (bool, optional): Aumenta a verbosidade da função. Por padrão é False.

        Returns:
            np.array: Lista de coeficientes da função característica.
        """    
        if isinstance(matrix,type(None)):
            matrix = self.A 
        base=-np.ones((1,matrix.shape[0]))[0]
        multi = np.linspace(matrix.shape[0],1,matrix.shape[0])
        tr = np.trace(matrix)
        det = np.linalg.det(matrix)
        res = np.concatenate((base**multi,[det]))
        res[1:-1] = res[1:-1]*tr
        res = -1*res if res[0]<0 else res
        return res

    def get_eigenval(self,matrix=None, verbose = False, debug = False ):
        """Calcula os autovalores relacionados à matriz de estados.

        Args:
            A (np.array[(n,n)]): Matriz de estados
            verbose (bool, optional): Aumenta a verbosidade da função. Por padrão é False.

        Returns:
            np.array: Lista de autovalores.
        """    
        if isinstance(matrix,type(None)):
            matrix = self.A 
        return np.roots(self.get_characteristic(matrix))

    def get_eigenvec(self,matrix = None , verbose = False, debug = False ):
        """Calcula os autovetores relacionados à matriz de estados.

        Args:
            A (np.array[(n,n)]): Matriz de estados
            verbose (bool, optional): Aumenta a verbosidade da função. Por padrão é False.

        Returns:
            np.array: Lista de autovetores.
        """    
        if(verbose or debug): print("+=== Eingenvector")
        if isinstance(matrix,type(None)):
            matrix = self.A 
        # Get eigenvalues
        eigval = self.get_eigenval(matrix)
    
        temp = np.tensordot(eigval,np.identity(matrix.shape[0]),0)-matrix
        temp_b = np.zeros((1,temp.shape[-1])).T
        
        eigvec = []
        for idx_a, a in enumerate(temp):
            if debug: print(f"  += Vector {idx_a}")
            normalized = np.array([b/b[0] for b in a])
            if debug: print("    Normalized = \n",normalized)
            li_comparison = np.all(np.array([np.isclose(normalized[0],b,rtol=1e-10) for b in normalized])==True)
            if debug: print("    Is L.I.?",li_comparison)
            if(li_comparison): 
                eig = np.array([[1],[-a[0,0]/a[0,1]]])
            else: 
                eig = np.array(np.linalg.solve(a,temp_b))
            eigvec.append(eig)
        eigvec = np.array(eigvec)
        return eigvec

    def get_system(self, A=None, X_init=None):
        """Calcula autovalores, autovetores e a matriz de amplitudes,
        com base na matriz de estados e nas condições iniciais.

        Args:
            A (np.array[(n,n)]): Matriz de estados
            X_init (np.array[(2,1)], optional): Matriz de condições iniciais. Por padrão 
            tem valor None, que inicializa as condições iniciais com valor unitário.

        Returns:
            np.array: Matriz de amplitudes.
        """    
        if isinstance(A,type(None)):
            A = self.A 
        if isinstance(X_init,type(None)):
            X_init = self.inital_contidition 
        eigenval =  self.get_eigenval(A)
        eigenvec =  self.get_eigenvec(A)
        if(np.any(X_init==None)): X_init = np.ones_like(eigenval)
        C = np.linalg.inv(np.concatenate(eigenvec,axis=1))@X_init
        return C

    def get_stability(self,  A=None, X_init=None,t=None,verbose=False):
        if isinstance(A,type(None)):
            A = self.A 
        if isinstance(X_init,type(None)):
            X_init = self.inital_contidition 
        autoval = self.get_eigenval(A)
        if verbose: print("Autovalor de A =",autoval,"\n")

        autovec = self.get_eigenvec(A)
        if verbose: print(f"Autovetor de A(Shape: {autovec.shape}) =\n{autovec}","\n")

        X_init = X_init#np.ones_like(autoval)
        if verbose: print(f"X =\n{X_init}","\n")

        C = self.get_system(A,X_init)
        if verbose: print(f"C =\n{C}","\n")

        X_dot=A@X_init
        if verbose: print(f"X_dot =\n{X_dot}","\n")

        pointc = self.get_pointcare_classification(A) # Sanity check
        if verbose: print(f"Pointcare classif.: {pointc}","\n")
        
        if isinstance(t,type(None)):
            t = np.linspace(0,1100,10000)

        X = np.array([C[n]*autovec[n]*np.exp(autoval[n]*t) for n,_ in enumerate(C)])
        X = np.sum(X,axis=0)
        X_0, X_1 = X[0],X[1]
        
        return t, X_0, X_1
        
    def plot_stability(self,  A=None, X_init=None ,dpi=120,save_path=None,plot=True):
        if isinstance(A,type(None)):
            A = self.A 
        if isinstance(X_init,type(None)):
            X_init = self.inital_contidition 
        t, X_0, X_1 = self.get_stability( A=A, X_init=X_init)

        plt.figure(dpi=dpi)
        plt.plot(t,X_0,label=f"$X_0$")
        plt.plot(t,X_1,label=f"$X_1$")
        plt.plot(t,X_0+X_1,label=f"$x(t)$",color="black")
        plt.xlabel("$t$")
        plt.ylabel("$X_n(t)$")
        plt.legend()
        plt.grid()

        if(save_path!=None and not(os.path.isdir(save_path))):
            save_path=None
        if save_path!=None:
            plt.savefig(os.path.join(save_path,"sis.png"))
            

        plt.figure(dpi=dpi)
        plt.plot(X_0, X_1, linewidth=1)
        plt.title(f"{self.pointcare_class}")
        plt.xlabel("$X_0$")
        plt.ylabel("$X_1$")
        plt.grid()
        if(save_path!=None and not(os.path.isdir(save_path))):
            save_path=None
        if save_path!=None:
            plt.savefig(os.path.join(save_path,"sis_stab.png"))
        
        if plot:
            plt.show()

    def get_pointcare_classification(self,  A=None):
        if isinstance(A,type(None)):
            A = self.A 
        T = np.trace(A)
        D = np.linalg.det(A)
        
        if(D<0): classif = "Sela"
        elif(T==0): classif = "Centro"
        elif(T**2>4*D): 
            if(T<0): classif = "Nó estável"
            else: classif = "Nó instável"
        else: 
            if(T<0): classif = "Foco estável"
            else: classif = "Foco instável"
        return classif

    def summary(self):
        print("+==========================================")
        print("+========== SUMMARY========================")
        print("+ A =\n",print_array(self.A))
        print("+ D = det(A) =",print_array(self.det))
        print("+ T = tr(A) =",print_array(self.trace))
        print("+ Characteristic: "," + ".join([f"{a_n:.4f}*lamb^{len(self.characteristic)-1-idx:d}" for idx, a_n in enumerate(self.characteristic)])+" = 0")
        print("+ Eingenvalues: \n",print_array(self.eigenval))
        print(f"+ Eingenvectors (Shape: {self.eigenvec.shape}):\n",print_array(self.eigenvec))
        print(f"+ X_init =\n",print_array(self.inital_contidition))
        print(f"+ C =\n",print_array(self.C))
        print(f"+ X_dot =\n",print_array(self.X_dot))
        print(f"+ Poincare classification:",self.pointcare_class)

        #print("==============")
        #print(tabulate(["a","b",self.A]))
        
        print("+==========================================")
        print("+==========================================")
        

def print_array(array, spaces=2):
    if isinstance(array,list):
        array = np.array(array)
    msg = "".join([" "*spaces+f' {s}\n' for s in np.array2string(array).split("\n")])
    return msg


def get_super_condition(a_1=None):
    # Condicao:
    #   a_1^2-4*a_0 > 0
    #     a_0 < a_1^2/4
    a_0 = (a_1**2)/4
    print(f'Superamortecido:\n  a_0 < {a_0:.3f}')
    return a_0

def get_crit_condition(a_1=None):
    # Condicao:
    #   a_1^2-4*a_0 = 0
    #     a_0 = a_1^2/4
    a_0 = a_1**2/4
    print(f'Criticamente amortecido:\n  a_0 = {a_0:.3f}')
    return a_0

def get_sub_condition(a_1=None):
    # Condicao:
    #   a_1^2-4*a_0 < 0
    #     a_0 > a_1^2/4
    a_0 = a_1**2/4
    print(f'Subamortecido:\n  a_0 > {a_0:.3f}')
    return a_0

def get_system_types(a_1=None):    
    a_0_sup = get_super_condition(a_1)
    a_0_crit = get_crit_condition(a_1)
    a_0_sub = get_sub_condition(a_1)
    return a_0_sup, a_0_crit, a_0_sub
    
def get_A(charac):
    charac=np.array(charac)
    A = np.identity(len(charac))
    
    if len(charac)<=2:
        A = np.fliplr(A)
    A[1]=-charac
    return A