# MCAP

MCAP: Monte Carlo adjusted profile.

## Installation

### From GitHub
```bash
pip install git+https://github.com/ChaosDonkey06/pyMCAProfile.git
```

### From source (local clone)
```bash
pip install .
```

### Editable install for development
```bash
pip install -e ".[dev]"
```

## Quick start

The distribution is named `pymcaprofile`; the import name is `mcap`.

```python
import numpy as np
from mcap import mcap_loglikelihood

# loglik: profile log-likelihood values evaluated at each value in `theta`
fit_df, mle_df = mcap_loglikelihood(loglik, theta, confidence=0.95, span=0.75)
print(mle_df)  # MLE, MC/statistical SE and confidence interval
```

## CLI

```bash
mcap --help
```
