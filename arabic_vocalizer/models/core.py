from pathlib import Path
from typing import Literal, List, Union

from ..models import Shakkala, Shakkelha

_MODEL_TYPE = Literal['shakkala', 'shakkelha']


def get_model(model: _MODEL_TYPE = 'shakkelha'):
    assert model in ('shakkala', 'shakkelha')

    data_folder = Path(__file__).parent.parent.joinpath('data')
    
    if model == 'shakkala':      
        return Shakkala(sd_path=data_folder.joinpath('shakkala.onnx').as_posix())     
    elif model == 'shakkelha':
        return Shakkelha(sd_path=data_folder.joinpath('shakkelha.onnx').as_posix()) 


def vocalize(input_text: Union[str, List[str]], 
          model: _MODEL_TYPE = 'shakkelha',
          return_probs: bool = False) -> Union[str, List[str]]:
    """
    Args:
        input_text
        model
        return_probs
    Returns:

    """
    assert model in ('shakkala', 'shakkelha')
    if not hasattr(vocalize, model):
        setattr(vocalize, model, get_model(model=model))

    if model == 'shakkala':
        return vocalize.shakkala.predict(input_text, return_probs=return_probs)
    elif model == 'shakkelha':
        return vocalize.shakkelha.predict(input_text, return_probs=return_probs)
    else:
        return  