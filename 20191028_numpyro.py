import torch

import numpy as onp


import jax.numpy as np
import jax.random as random
from jax.scipy.special import logsumexp



import numpyro
import numpyro.distributions as dist

from numpyro.infer import MCMC, NUTS, log_likelihood, predictive

from Bio.PDB import *
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import time

import multiprocessing



n=14


def save_M(M, f_out):
    """
    Save CA trace of M in PDB file f_out.
    """
    _ATOM = '%s%5i  %-4s%3s %c%4i%c   %8.3f%8.3f%8.3f%6.2f%6.2f %4s%2s%2s\n'

    def get_ATOM_line(atom_i, name, resid, x, y, z, aa_type):
        """
        Write PDB ATOM line.
        """
        args=('ATOM  ', atom_i, name, aa_type, 'A', resid, ' ', x, y, z, 0.0, 0.0, 'X', ' ', ' ')
        s = _ATOM % args
        return s

    fp = open(f_out, 'w')
    for i in range(0, M.shape[0]):
        x, y, z = M[i]
        s = get_ATOM_line(i, 'CA', i, x, y, z, 'ALA') 
        fp.write(s)
    fp.close()

def get_samples(posterior, name):
    """
    Extracts samples from a posterior object.
    """
    marginal = posterior.marginal(sites=[name])
    marginal_tensor = marginal.support()[name]
    return marginal_tensor

def get_CA_coords(protein_name, n):
    """
    Gets coordinates of nth CA atom. 
    """
    # Get protein structure
    parser = PDBParser()
    struct = parser.get_structure(protein_name, protein_name + '.pdb')
    # Get the coordinates of CA atoms
    coords = []
    for model in struct:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == 'CA':
                        XYZ = atom.get_coord()
                        coords.append(XYZ)
    # Get the nth atom coordinates
    nth_coord = coords[n]
    return nth_coord


# Creating a list of first native protein 10 amino acids CA atoms coordinates
native_coords = [] 
for i in range(n):
    native_coords.append(get_CA_coords('../loop_1ttj', i))
native_coords_t = torch.tensor(native_coords)

#native_coords_t
# Find first 3 coordinates of native_coords_t
#first3 = torch.zeros([3,3])
first3 = torch.zeros([3,3])
for i in range(3):
    first3[i] = native_coords_t[i]


M_first=np.array(first3)





dist_nr=[]
points=[(0,13),(4,9),(2,10),(0,8),(6,13)]

for p in points:
    
    d=torch.dist(native_coords_t[p[0]],native_coords_t[p[1]])
    
    dist_nr.append(d)
    


def rmsd_dist_burn(burn,s_b,s_d,distances=1, sample_nr=10, target_accept_prob=0.4):
    """
    The function runs NUTS sampler based on the specific model for sampling protein 
    structure with given pairwise distances.
    distance: number of random distances to be additionally restraint;
    burn: warm up size;
    sample_nr: number of samples to run algorithm;
    target_accept_prob: target acceptance probability, NUTS Sampler parameter;
    
    Returns: all structure average RMSD and separate values, fixed 3 first coordinates 
    average RMSD and separate values, time that it took each iteration to run.
    """
    rmsd_all = []
    rmsd_first3 = []
    times = []
    
    def model(N=n):
        plate1=numpyro.plate("aa", N-3, dim=-2)
        plate2=numpyro.plate("coord", 3, dim=-1)
        with plate1, plate2:
            M_last = numpyro.sample("M", dist.StudentT(1, 0, 50))      
        
        #M_last = numpyro.sample('M', dist.Normal(0, 10).expand_by([N-3,3]).to_event(1)) 

        # Stack fixed and moving coordinates

        M=np.concatenate((M_first, M_last))
        
        # Make sure bond distances are around 3.8 Å       
        bonds=M[0:-1]- M[1:]
        
        Bonds=(bonds[:,0]**2+bonds[:,1]**2+bonds[:,2]**2)**(1/2)

#        for i in pyro.plate('bonds', N-1):
#            bond=Bonds[i]
#            bond_obs = pyro.sample('bond_%i' % i, dist.Normal(bond, 0.001), obs=torch.tensor(3.8))

        i=0       
        with numpyro.plate("Bonds",13):
            bond_obs=numpyro.sample("Bonds_%i" % i, dist.Normal(Bonds, 0.001), obs=3.8)
            i+=1
 
        # Add a distance restraint between first and last point
        
        D = M[0] -  M[-1]
        d = (D[0]**2+D[1]**2+D[2]**2)**(1/2)
        
        d_obs = numpyro.sample("d_obs", dist.Normal( d, 0.001), obs=(dist_nr[0].item()))
        
        for i in range(1,distances):
            D = (M[points[i][0]] - M[points[i][1]])
            d = (D[0]**2+D[1]**2+D[2]**2)**(1/2)
            d_obs=numpyro.sample('d%s_obs' % i, dist.Normal(d, s_d), obs=(dist_nr[i].item()))               
    
    # Nr samples
    S=1000
    # Nr samples burn-in
    B=burn
    # Do NUTS sampling
    nuts_kernel = NUTS(model, adapt_step_size=True, target_accept_prob=target_accept_prob)
    mcmc_sampler = MCMC(nuts_kernel,B, num_samples=S)
    
    rng= random.PRNGKey(1)
    
    posterior = mcmc_sampler.run(rng)
    # Get the last sampled points
    #samples = get_samples(posterior, 'M')
    M_last=mcmc_sampler.get_samples()
    M=np.concatenate((M_first, M_last['M'][-1]))


    #M=samples[S-1]
    #M=torch.cat((first3, M_last))  # Add fixed first 3 coordinates
    

    # or return samples for pdb file:
    return M# M['M'][-1]



def model_check(M,distances=1):    
    M=torch.tensor(onp.array((M)))
    #Check that bound distance is 3.8 Å    
    bounds=[]
    for i in range(n-1):
        bound=torch.dist(M[i],M[i+1]).item()
        #print(bound)
        bounds.append(bound)
    rmsd_b=0
    for i in range(n-1):
        rmsd_b += (bounds[i]-3.8)**2
    rmsd_b=math.sqrt(rmsd_b/(n-1))
    
    rmsd_d=0
    N=0
    for i in range(n):
        for j in range(i+1,n):
            a,b=i,j
            d_M=round(torch.dist(M[a],M[b]).item())
            d_n=round(torch.dist(native_coords_t[a],native_coords_t[b]).item())
            rmsd_d+=(d_M-d_n)**2
            N+=1
            if (a,b) in points[:distances] or (a,b) in points[:distances]:
                print('Restricted distance ',str(a),',',str(b),',',(d_M-d_n))
            else:
                print('Not restricted distance ',str(a),',',str(b),',',(d_M-d_n))
    print(N)
    rmsd_d=math.sqrt(rmsd_d/(N))

    return rmsd_b,rmsd_d



#d=int((n*(n-1))/2)
d=5
def fun(s_d,s_b):
#    print(s_d,s_b)
    name='d_'+str(d)+'_sd_'+str(s_d)+'_sb_'+str(s_b)
    M = rmsd_dist_burn(distances=d, sample_nr=1, burn=70,s_d=s_d,s_b=s_b,target_accept_prob = 0.4)

    rmsd_b,rmsd_d=model_check(M,d)
    #print(rmsd_b,rmsd_d)
    save_M(M,'../20191027_npy_'+name+'.pdb')
 
    return 



fun(s_b=0.001,s_d=0.001)





