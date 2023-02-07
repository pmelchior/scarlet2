import jax.random as jr
import equinox as eqx
from models_eqx import ScoreNet

# model params
patch_size=4
hidden_size=64
mix_patch_size=512
mix_hidden_size=512
num_blocks=4
t1=10.0

# noise -- time schedule
int_beta = lambda t: t  

# rng numbers 
key = jr.PRNGKey(42)
model_key, train_key, loader_key, sample_key = jr.split(key, 4)

# initialise model for 64 res
data_shape = (1, 64, 64)
model64 = ScoreNet(
        data_shape,
        patch_size,
        hidden_size,
        mix_patch_size,
        mix_hidden_size,
        num_blocks,
        t1,
        key=model_key,
)

# load 64 res model
SAVE_DIR = 'stored_models'
fn = SAVE_DIR + '/eqx_scorenet64.eqx'
ScoreNet64 = eqx.tree_deserialise_leaves(fn, model64)

# initialise model for 32 res
data_shape = (1, 32, 32)
model32 = ScoreNet(
        data_shape,
        patch_size,
        hidden_size,
        mix_patch_size,
        mix_hidden_size,
        num_blocks,
        t1,
        key=model_key,
)

# load 32 res model
SAVE_DIR = 'stored_models'
fn = SAVE_DIR + '/eqx_scorenet32.eqx'
ScoreNet32 = eqx.tree_deserialise_leaves(fn, model32)
