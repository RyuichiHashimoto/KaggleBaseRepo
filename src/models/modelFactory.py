from copy import deepcopy

from models.modelBase import ModelBase, ModelParameter


class ModelFactory:
    """
    A factory class for managing the creation and registration of ModelBase instances using associated ModelParameter subclasses.

    The ModelFactory allows you to associate specific ModelParameter subclasses with ModelBase subclasses for
    dynamic model creation. This simplifies the process of model instantiation, by encapsulating the
    creation logic within the factory.

    ModelFactory is designed to be used as a decorator in the definition of ModelBase subclasses,
    associating them with a corresponding ModelParameter subclass.

    Example:
        # Define your ModelParameter subclass
        @dataclass
        class MLP_Parameter(ModelParameter):
            layer_sizes: List[int]
            activation: str
            solver: str

        # Register your ModelBase subclass with its associated ModelParameter
        @ModelFactory.register(MLP_Parameter)
        class MLP_Model(ModelBase):
            ...

    Usage:
        # Create a new model by passing a ModelParameter instance to the factory's create method
        params = MLP_Parameter(layer_sizes=[10, 10], activation='relu', solver='adam')
        model = ModelFactory.create(params)

    Notes:
        - The ModelBase and ModelParameter classes are not defined in this code snippet and should be imported from their respective modules.
        - The model classes (like MLP_Model in this example) should also be defined appropriately.

    Attributes:
        registry: A dictionary that maps ModelParameter class names to their associated ModelBase classes.

    Methods:
        get_registories: Returns a deep copy of the registry dictionary.
        register: Registers a ModelBase class in the registry against an associated ModelParameter subclass.
        create: Creates an instance of a ModelBase subclass using an instance of the associated ModelParameter subclass.
    """

    __registry: dict[str, ModelBase] = {}

    @classmethod
    def get_registories(cls) -> dict[str, ModelBase]:
        return deepcopy(cls.__registry)  # use deepcopy function to avoid shallow copy

    @classmethod
    def register(cls, modelParameter: ModelParameter):
        def inner_wrapper(modelClass):
            print(modelParameter.__name__)
            if not issubclass(modelClass, ModelBase):
                raise TypeError(f"Registered class {modelClass.__name__} is not a subclass of ModelBase")
            if not issubclass(modelParameter, ModelParameter):
                raise TypeError(f"Registered class {modelParameter.__name__} is not a subclass of ModelParameter")

            if modelParameter.__name__ in cls.__registry:
                raise ValueError(
                    f"The class '{modelParameter.__name__}' has already been registered. Duplicate registration is not allowed."
                )

            cls.__registry[modelParameter.__name__] = modelClass
            return modelClass

        return inner_wrapper

    @classmethod
    def create(cls, parameter: ModelParameter):
        cls_ = cls.__registry[parameter.__class__.__name__]
        return cls_(parameter)
