# model_factory.py
import importlib

def create_model(model_config: dict):
    """
    Creates a model instance based on the configuration.
    Assumes model class definition is in 'model.py'.

    Args:
        model_config (dict): Dictionary containing model parameters,
                              including 'name' for the class name.

    Returns:
        torch.nn.Module: An instance of the specified model.
    """
    model_name = model_config.get('name')
    if not model_name:
        raise ValueError("Model configuration must include a 'name' key.")

    try:
        model_module = importlib.import_module("model")
        ModelClass = getattr(model_module, model_name)
    except ImportError:
        raise ImportError("Could not import 'model.py'. Make sure it exists and is in the Python path.")
    except AttributeError:
        raise AttributeError(f"Model class '{model_name}' not found in 'model.py'.")

    model_args = {k: v for k, v in model_config.items() if k != 'name'}

    print(f"Creating model: {model_name} with args: {model_args}")
    model = ModelClass(**model_args)
    return model