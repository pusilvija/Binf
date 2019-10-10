
# coding: utf-8

# In[ ]:


# %load m_springs.py
import pyro
import torch
import pyro.distributions as dist
import torch.nn.functional as func
from pyro.infer.mcmc import MCMC, NUTS




def save_M(M, f_out):
    """
    Save CA trace of M in PDB file f_out.
    (it adds - poorly placed - N and C atoms between the CA atoms)
    """
    _ATOM = "%s%5i  %-4s%3s %c%4i%c   %8.3f%8.3f%8.3f%6.2f%6.2f %4s%2s%2s\n"

    def get_ATOM_line(atom_i, name, resid, x, y, z, aa_type):
        """
        Write PDB ATOM line.
        """
        args=("ATOM  ", atom_i, name, aa_type, "A", resid, " ", x, y, z, 0.0, 0.0, "X", " ", " ")
        s = _ATOM % args
        return s

    fp = open(f_out, "w")

    labels=["CA", "C", "N"]
    for i in range(0, M.shape[0]-1):
        v1=M[i]
        v2=M[i+1]
        delta=(v2-v1)/3.0
        for j in range(0,3):
            # Add C and N atoms in between the CA
            x, y, z = v1+j*delta
            s = get_ATOM_line(i*3+j, labels[j], i+1, x, y, z, "ALA")
            fp.write(s)
    # Last CA
    x,y,z=M[-1]
    s=get_ATOM_line(3*(M.shape[0]-1), "CA", M.shape[0], x, y, z, "ALA")
    fp.write(s)
    fp.close()



def get_samples(posterior, name):
    """
    Extracts samples from a posterior object.
    """
    marginal = posterior.marginal(sites=[name])
    marginal_tensor = marginal.support()[name]
    return marginal_tensor

# Fix first three coordinates
M_first=torch.tensor(((0,0,0),
                      (3.8,0,0),
                      (3.8,3.8,0.0)))

def model(N=10):
    # Sample N-3 random points according to a Normal distribution 
    # The plates render all the coordinates independent
    plate1=pyro.plate("aa", N-3, dim=-2)
    plate2=pyro.plate("coord", 3, dim=-1)
    with plate1, plate2:
        M_last = pyro.sample("M", dist.Normal(0, 20))

    # Stack fixed and moving coordinates
    M=torch.cat((M_first, M_last))

    # Make sure bond distances are around 3.8 Å
    # Standard deviation of bonds
    
    #sb=pyro.sample("sigma_bond", dist.HalfCauchy(scale=0.1))

    # Calculate bond distances
    # (skip first two bonds, as they are fixed)
    bonds=torch.dist(M[2:-1], M[3:])
    with pyro.plate("bonds"):
        bond_obs=pyro.sample("bonds", dist.StudentT(1, bonds, 0.001), obs=torch.tensor(3.8))

    # Add a distance restraint between first and last point
    # Standard deviation of pairwise distance
    sd=pyro.sample("sigma_dist", dist.HalfCauchy(scale=0.1))
    d = torch.dist(M[0], M[-1])
    d_obs = pyro.sample("d_obs", dist.StudentT(1, d, 0.001), obs=torch.tensor(10))

filename_pdb=

if __name__=="__main__":
    # Nr samples
    S= 500
    # Nr samples burn-in
    B=70

    # Do NUTS sampling
    nuts_kernel = NUTS(model, adapt_step_size=True)
    mcmc_sampler = MCMC(nuts_kernel, num_samples=S, warmup_steps=B)
    posterior = mcmc_sampler.run()

    # Get the last sampled points
    samples = get_samples(posterior, "M")
    # Save to PDB file
    M_last=samples[S-1]
    M=torch.cat((M_first, M_last))  # Add fixed first 3 coordinates
    save_M(M, filname_pdb+".pdb")

