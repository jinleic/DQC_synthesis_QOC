from bqskit import Circuit
from bqskit.compiler.machine import MachineModel
from typing import Any, Iterator
import itertools as it
import copy

class PassData:
    _reserved_keys = [
        'target',
        'model',
        'placement',
        'error',
        'seed',
        'machine_model',
        'initial_mapping',
        'final_mapping',
    ]
    def __init__(self,
                 circuit: Circuit,
                 ):
        self._target = circuit.get_unitary()
        self._error = 0.0
        self._model = MachineModel(circuit.num_qudits)
        self._placement = list(range(circuit.num_qudits))
        self._initial_mapping = list(range(circuit.num_qudits))
        self._final_mapping = list(range(circuit.num_qudits))
        self._data = {}
        self._seed = None
    
    @property
    def target(self):
        return self._target
    
    @target.setter
    def target(self, _val) -> None:
        if len(self.placement) != _val.num_qudits:
            self.placement = list(range(_val.num_qudits))
        self._target = _val

    @property
    def error(self) -> float:
        """Return the current target unitary or state."""
        return self._error

    @error.setter
    def error(self, _val: float) -> None:
        self._error = _val

    @property
    def model(self) -> MachineModel:
        """Return the current target MachineModel."""
        return self._model

    @model.setter
    def model(self, _val: MachineModel) -> None:
        if not isinstance(_val, MachineModel):
            raise TypeError(
                f'Cannot set model to {type(_val)}.'
                ' Expected a MachineModel.',
            )

        self._model = _val

    @property
    def gate_set(self):
        """Return the current target MachineModel's GateSet."""
        return self._model.gate_set

    @gate_set.setter
    def gate_set(self, _val) -> None:
        self._model.gate_set = _val

    @property
    def placement(self):
        """Return the current placement of circuit qudits on model qudits."""
        return self._placement

    @placement.setter
    def placement(self, _val) -> None:
        self._placement = list(int(x) for x in _val)

    @property
    def initial_mapping(self):
        """
        Return the initial mapping of logical to physical qudits.

        This always maps how the logical qudits from the original circuit start
        on the physical qudits of the current circuit.
        """
        return self._initial_mapping

    @initial_mapping.setter
    def initial_mapping(self, _val) -> None:

        self._initial_mapping = list(int(x) for x in _val)

    @property
    def final_mapping(self):
        """
        Return the final mapping of logical to physical qudits.

        This always maps how the logical qudits from the original circuit end on
        the physical qudits of the current circuit.
        """
        return self._final_mapping

    @final_mapping.setter
    def final_mapping(self, _val) -> None:

        self._final_mapping = list(int(x) for x in _val)

    @property
    def seed(self):
        """Return the pass's seed."""
        return self._seed

    @seed.setter
    def seed(self, _val) -> None:
        self._seed = _val

    @property
    def connectivity(self):
        """Retrieve the physical connectivity of the circuit qudits."""
        return self.model.coupling_graph.get_subgraph(self.placement)

    def __getitem__(self, _key: str) -> Any:
        """Retrieve the value associated with `_key` from the pass data."""
        if _key in self._reserved_keys:
            if _key == 'machine_model':
                _key = 'model'
            return self.__getattribute__(_key)

        return self._data.__getitem__(_key)

    def __setitem__(self, _key: str, _val: Any) -> None:
        """Update the value associated with `_key` in the pass data."""
        if _key in self._reserved_keys:
            if _key == 'machine_model':
                _key = 'model'
            return self.__setattr__(_key, _val)

        return self._data.__setitem__(_key, _val)

    def __delitem__(self, _key: str) -> None:
        """Delete the key-value pair associated with `_key`."""
        if _key in self._reserved_keys:
            raise RuntimeError(f'Cannot delete {_key} from data.')

        return self._data.__delitem__(_key)

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over all keys in the pass data."""
        return it.chain(self._reserved_keys.__iter__(), self._data.__iter__())

    def __len__(self) -> int:
        """Return the number of key-value pairs in the pass data."""
        return self._data.__len__() + len(self._reserved_keys)

    def __contains__(self, _o: object) -> bool:
        """Return true if `_o` is a key in the pass data."""
        in_resv = self._reserved_keys.__contains__(_o)
        in_data = self._data.__contains__(_o)
        return in_resv or in_data

    def copy(self) :
        """Returns a deep copy of the data."""
        return copy.deepcopy(self)

    def become(self, other, deepcopy: bool = False) -> None:
        """Become a copy of `other`."""
        if deepcopy:
            self._target = copy.deepcopy(other._target)
            self._error = copy.deepcopy(other._error)
            self._model = copy.deepcopy(other._model)
            self._placement = copy.deepcopy(other._placement)
            self._data = copy.deepcopy(other._data)
            self._seed = copy.deepcopy(other._seed)
        else:
            self._target = copy.copy(other._target)
            self._error = copy.copy(other._error)
            self._model = copy.copy(other._model)
            self._placement = copy.copy(other._placement)
            self._data = copy.copy(other._data)
            self._seed = copy.copy(other._seed)

    def update_error_mul(self, error: float) -> None:
        """Update the error multiplicatively."""
        self.error = (1 - ((1 - self.error) * (1 - error)))
