# Configuration

Configuration for the data set.

Decided to go for a 'Pydantic' data class. It's slightly more complicated that just having yaml
files, but the 'Pydantic' feature I think outweigh this. There is a load from yaml function also.

See `model.py` for documentation of the expected configuration fields.

All paths must include the protocol prefix. For local files, it's sufficient to just start with a
'/'. For aws, start with 's3://', for gcp start with 'gs://'.

# Example

```python
# import the load function
from ocf_data_sampler.config.load import load_yaml_configuration

# load the configuration
configuration = load_yaml_configuration(filename)
```
