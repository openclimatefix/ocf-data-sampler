import os
import pickle


class PickleCacheMixin:
    def __init__(self, *args, **kwargs):
        self._pickle_path = None
        super().__init__(*args, **kwargs)  # cooperative multiple inheritance

    def presave_pickle(self, pickle_path: str) -> None:
        """Save the full object state to a pickle file and store the pickle path."""
        self._pickle_path = pickle_path
        with open(pickle_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def __getstate__(self):
        """If presaved, only pickle reference. Otherwise pickle everything."""
        if self._pickle_path:
            return {"_pickle_path": self._pickle_path}
        else:
            return self.__dict__

    def __setstate__(self, state):
        """Restore object from pickle, reloading from presaved file if possible."""
        self.__dict__.update(state)
        if self._pickle_path and os.path.exists(self._pickle_path):
            with open(self._pickle_path, "rb") as f:
                saved_state = pickle.load(f)
                self.__dict__.update(saved_state)