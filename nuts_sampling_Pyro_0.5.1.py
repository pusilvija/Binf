if __name__=="__main__":
    # Nr samples
    S=1000
    # Nr samples burn-in
    B=50
    
    # Do NUTS sampling
    nuts_kernel = NUTS(model, adapt_step_size=True, target_accept_prob=0.4)
    mcmc_sampler = MCMC(nuts_kernel, num_samples=S, warmup_steps=B)
    mcmc_sampler.run()
    # Get the last sampled points
    samples = mcmc_sampler.get_samples()['M']
    # Save to PDB file
    M_last=samples[-1]
    M=torch.cat((M_first, M_last))  # Add fixed first 3 coordinates
    save_M(M, "random.pdb")
