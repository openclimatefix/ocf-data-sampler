"""A mixin to handle pickling and caching a dataset's state."""

import os
import pickle


class PickleCacheMixin:
    """A mixin for classes that need to cache their state using pickle."""
    def __init__(self, *args: list, **kwargs: dict) -> None:
        """Initialize the pickle path and call the parent constructor."""
        self._pickle_path = None
        super().__init__(*args, **kwargs)  # cooperative multiple inheritance

    def presave_pickle(self, pickle_path: str) -> None:
        """Save the full object state to a pickle file and store the pickle path."""
        self._pickle_path = pickle_path
        with open(pickle_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def __getstate__(self) -> dict:
        """If presaved, only pickle reference. Otherwise pickle everything."""
        if self._pickle_path:
            return {"_pickle_path": self._pickle_path}
        else:
            return self.__dict__

    def __setstate__(self, state: dict) -> None:
        """Restore object from pickle, reloading from presaved file if possible."""
        self.__dict__.update(state)
        if self._pickle_path and os.path.exists(self._pickle_path):
            with open(self._pickle_path, "rb") as f:
                saved_state = pickle.load(f) # noqa: S301
                self.__dict__.update(saved_state)
