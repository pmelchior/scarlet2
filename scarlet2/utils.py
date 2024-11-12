# utility functions for Scarlet2
import numpy as np
import jax.numpy as jnp
from scarlet2 import *


 # --------------------- # 
 # Hessian approximation # 
 # --------------------- # 
 # https://arxiv.org/pdf/2006.00719.pdf 
  
 # for regular functions f 
 def hvp(f, primals, tangents): 
     return jvp(grad(f), primals, tangents)[1] 
  
 # for score functions 
 def hvp_grad(grad_f, primals, tangents): 
     return jvp(grad_f, primals, tangents)[1] 
  
 # diagonals of Hessian from HVPs 
 def hvp_rad(hvp, shape): 
     max_iters = 100 # maximum number of iterations 
     H = jnp.zeros(shape, dtype=jnp.float32) 
     H_ = jnp.zeros(shape, dtype=jnp.float32) 
     for i in range(max_iters): 
         key = random.PRNGKey(i) 
         z = random.rademacher(key, shape , dtype=jnp.float32) 
         H += jnp.multiply(z, hvp(z)) 
         if i > 0: 
             norm = jnp.linalg.norm(H/(i+1) - H_/i, ord=2) 
             if norm < 1e-6 * jnp.linalg.norm(H/(i+1), ord=2): # gets reasonable results with 1e-2 
                 break 
         H_ = H 
     return H/(i+1)
