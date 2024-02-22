import jax
import optax
import numpy as np
import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping

import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as sharding

import functools as ft
def crop(x, y):

    _,hx,wx = x.shape
    _,hy,wy = y.shape

    hs = (hx-hy)//2
    ws = (wx-wy)//2

    return x[:,
    hs:-(hx-hs-hy),
    ws:-(wx-ws-wy)]

def pad_it(x, size=104):
    """Zero-pads the input image to the model size
    
    Parameters
    ----------
    x : array of the bounding box
    size: int
        size of the model to be used
        
    Returns
    -------
    x : padded array same size as model_size
    """
    data_size = x.shape[1]
    pad = True
    pad_gap = size - data_size
    assert pad_gap>= 0, "Model size must be larger than max box size"
    # dont pad if we dont need to
    if pad_gap == 0:
        pad = False
    # calculate how much to pad    
    if pad_gap % 2 == 0:
        pad_lo = pad_hi = int(pad_gap / 2)
    else:
        pad_lo = int(pad_gap // 2)
        pad_hi = int(pad_lo + 1)
    # perform the zero-padding
    if pad:
        if jnp.ndim(x) == 3:
            x = jnp.pad(x, ((0,0),(pad_lo, pad_hi), (pad_lo, pad_hi)), 'constant', constant_values=0) 
        else:
            x = jnp.pad(x, ((pad_lo, pad_hi), (pad_lo, pad_hi)), 'constant', constant_values=0)
    return x

class UNet(eqx.Module):
    enc_layers: list
    dec_layers: list
    outconv:    eqx.Module
        
    def __init__(self, key, encChannels, decChannels):
        
        self.enc_layers = []
        self.dec_layers = []
        padding = 1
        stride  = 1
        
        # encoder parts
        key1, key2 = jax.random.split(key, 2)
        for i in range(len(encChannels) - 1):
            
            e1 = eqx.nn.Conv2d(in_channels=encChannels[i],  out_channels=encChannels[i+1], 
                               kernel_size = 3, key = key1, stride=stride, padding = padding)
            e2 = eqx.nn.Conv2d(in_channels=encChannels[i+1],  out_channels=encChannels[i+1], 
                               kernel_size = 3, key = key2, stride=stride, padding = padding)
            p1 = eqx.nn.MaxPool2d(kernel_size=2, stride=2) 
            
            self.enc_layers.append([e1,e2,p1])
            key1, key2 = jax.random.split(key1, 2)
            
        # decoder parts
        key1, key2, key3 = jax.random.split(key1, 3)
        for i in range(len(decChannels) - 1):
            u1 = eqx.nn.ConvTranspose2d(
                                    in_channels  = decChannels[i], 
                                    out_channels = decChannels[i + 1],
                                    kernel_size  = 2,
                                    stride       = 2,
                                    key          = key1)
            d1 = eqx.nn.Conv2d(in_channels=decChannels[i],  out_channels=decChannels[i+1], 
                               kernel_size = 3, key = key2, stride=stride, padding = padding)
            d2 = eqx.nn.Conv2d(in_channels=decChannels[i+1],  out_channels=decChannels[i+1], 
                               kernel_size = 3, key = key3, stride=stride, padding = padding)

            self.dec_layers.append([u1,d1,d2])
            key1, key2, key3 = jax.random.split(key1, 3)
            
        # Output layer
        self.outconv = eqx.nn.Conv2d(in_channels=decChannels[-1], out_channels=2, kernel_size=1, key=key1)
        
    def __call__(self, x):
        
        x0 = x; tracker = []
        
        # Encoder steps
        for l in self.enc_layers[:-1]:
            x0 = jax.nn.relu(l[0](x0))
            x0 = jax.nn.relu(l[1](x0)); tracker.append(x0)
            x0 = l[2](x0)

        x0 = jax.nn.relu(self.enc_layers[-1][0](x0))
        x0 = jax.nn.relu(self.enc_layers[-1][1](x0))

        # Decoding steps
        for l,tracked in zip(self.dec_layers,tracker[::-1]):
            
            x0 = l[0](x0)
            x0 = jnp.concatenate([x0,tracked], axis=0)
            # x0 = jnp.concatenate([x0,crop(tracked,x0) ], axis=0)

            x0 = jax.nn.relu(l[1](x0))
            x0 = jax.nn.relu(l[2](x0))

        # Output layer
        x0 = self.outconv(x0)

        return x0

def loss(model, x, y, w, s, combine = True):
    li = 0; ls = 0
    for xi, yi, wi, si in zip(x,y,w,s):
        pred_y = jax.vmap(model)(xi)
        li += cross_entropy(yi, pred_y[:,:1,:,:], wi)
        ls += jnp.sum(
                jnp.power( (si-pred_y[:,1:,:,:])/si,2)
                ) / 1e5

    if combine:
        return (li+ls) / (len(x[0]) * len(x))
    else:
        return li / (len(x[0]) * len(x)), ls / (len(x[0]) * len(x))

def cross_entropy(
    y, pred_y, w
):
    return jnp.sum(w*(pred_y-y)**2)

def train(
    model: UNet,
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    Sigma: np.ndarray,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
    batch_size:  int,
    ngroup: int,
):
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: UNet,
        opt_state: PyTree,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        sigma: np.ndarray,
    ):
        print(x.shape)
        x,y,w,sigma = jnp.array(np.split(x,ngroup)), jnp.array(np.split(y,ngroup)), jnp.array(np.split(w,ngroup)), jnp.array(np.split(sigma,ngroup))
        print(x.shape)

        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y, w, sigma)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    import timeit
    ts = timeit.default_timer()

    S  = []; L = []
    for step, (x, y, w, sigma) in zip(range(steps), dataloader((X, Y, W, Sigma), batch_size)):

        model, opt_state, train_loss = make_step(model, opt_state, x, y, w, sigma)

        
        S.append(step); L.append(train_loss)
        if (step % print_every) == 0 or (step == steps - 1):
            x,y,w,sigma = jnp.array(np.split(x,ngroup)), jnp.array(np.split(y,ngroup)), jnp.array(np.split(w,ngroup)), jnp.array(np.split(sigma,ngroup))

            ls,li = loss(
                model, x, y, w, sigma, combine = False
            )
            print(
                "step={}, train_loss={:.4},{:.4}, ".format(step,ls,li) +'time = {:.4}min'.format( 
                    (timeit.default_timer() - ts)/60 
                 )
            )
            ts = timeit.default_timer()
            
    return model, np.array([S,L]).T

def transform(x,alpha=0.1,beta=1,inverse=False):
    if inverse:
        x = jnp.sinh(x*beta)* alpha
    else:
        x = jnp.arcsinh(x/alpha)/beta
    return x

def vgrad(f,x):
    y, vjp_fn = vjp(f,x)
    return vjp_fn(jnp.ones(y.shape))[0]

def dataloader(arrays, batch_size):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = np.arange(dataset_size)
    while True:
        perm = np.random.permutation(indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size