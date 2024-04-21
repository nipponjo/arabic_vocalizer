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
             return_probs: bool = False
             ) -> Union[str, List[str]]:
    """
    Parameters:
        input_text (str|list[str]): Unvocalized text
        model: Vocalization model [shakkala|shakkelha]
        return_probs: Return probabilities?
        
    Returns:
        (str|list[str]): Predicted vocalized text
    
    Examples:
        >>> from arabic_vocalizer import vocalize
        >>> input_text = "اللغة العربية هي أكثر اللغات السامية تحدثا، وإحدى أكثر اللغات انتشارا في العالم، يتحدثها أكثر من 467 مليون نسمة"
        # shakkala output
        >>> print(vocalize(input_text, model='shakkala'))
        >>> اللُّغَةُ الْعَرَبِيَّةُ هِيَ أَكْثَرُ اللُّغَاتِ السَّامِيَةِ تَحَدُّثًا، وَإِحْدَى أَكْثَرِ اللُّغَاتِ انْتِشَارًا فِي الْعَالِمِ، يَتَحَدَّثُهَا أَكْثَرُ مَنْ 467 مُلْيُونُ نُسْمَةَ
        # shakkelha output
        >>> print(vocalize(input_text, model='shakkelha'))
        >>> اللُّغَةُ الْعَرَبِيَّةُ هِيَ أَكْثَرُ اللُّغَاتِ السَّامِيَةِ تَحَدُّثًا، وَإِحْدَى أَكْثَرِ اللُّغَاتِ انْتِشَارًا فِي الْعَالِمِ، يَتَحَدَّثُهَا أَكْثَرُ مِنْ 467 مَلْيُونٍ نَسَمَةً

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