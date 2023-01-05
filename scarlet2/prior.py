import distrax

class Prior(distrax.Distribution):
    pass

class NoPrior(distrax.Uniform):
    def log_prob(self, x):
        return 0.
