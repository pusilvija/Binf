

def model_check():
    
    #Check that 3 first coords are equal
    for i in range(3):
        if (torch.all(torch.eq(M[i,:], native_coords_t[i,:]))).item()!=1:
            print('coord %i' % i , 'not equal to native')
    
    #Check that bound distance is 3.8 Å
    for i in range(N-1):
        bound=round(torch.dist(M[i],M[i+1]).item(),1)
        if bound != 3.8:
            print('bound %i' % i ,i+1, 'not 3.8, is ',bound)
    #Check distance btw first and last
    
    d=round(torch.dist(M[0],M[N-1]).item())
    d_n=round(torch.dist(native_coords_t[0],native_coords_t[N-1]).item())
    if d != d_n:
        print('distance btw first and last not equal to native(%i)',d,d_n)
        #Check restricted distances
    for i in range(N):
        for j in range(i+1,N):
            a,b=i,j
            d_M=round(torch.dist(M[a],M[b]).item())
            d_n=round(torch.dist(native_coords_t[a],native_coords_t[b]).item())
            if d_M != d_n:
                print('distance between' ,a ,'and %i not correct' % b,d_M,d_n)
