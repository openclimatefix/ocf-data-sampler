"""A mixin for classes that need to cache their state using pickle."""
import os
import pickle


class PickleCacheMixin:
    """A mixin for classes that need to cache their state using pickle."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize the pickle path and call the parent constructor."""
        self._pickle_path = None
        super().__init__(*args, **kwargs)  # cooperative multiple inheritance

    def presave_pickle(self, pickle_path: str) -> None:
        """Save the full object state to a pickle file and store the pickle path.

        The object will be pickled by reference after calling this function, so the 
        file must be readable wherever it is unpickled. 
        
        The saved state is a snapshot - call this again after mutating the object.

        Args:
            pickle_path: Where to write the object state to.
        """
        self._pickle_path = pickle_path
        with open(pickle_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def __getstate__(self) -> dict:
        """If presaved, only pickle reference. Otherwise pickle everything."""
        if self._pickle_path:
            return {"_pickle_path": self._pickle_path}
        else:
            # Copied so that the pickled state can't be mutated through the live object
            return dict(self.__dict__)

    def __setstate__(self, state: dict) -> None:
        """Restore object from pickle, reloading from presaved file if possible."""
        self.__dict__.update(state)

        if not self._pickle_path:
            return

        if not os.path.exists(self._pickle_path):
            raise FileNotFoundError(
                f"Presaved state file not found: {self._pickle_path}. This object was pickled by "
                "reference to that path - see `presave_pickle`.",
            )

        with open(self._pickle_path, "rb") as f:
            saved_state = pickle.load(f)  # noqa: S301

        self.__dict__.update(saved_state)
