Arabic deep-learning based diacritization models ([Shakkala](https://github.com/Barqawiz/Shakkala), [Shakkelha](https://github.com/AliOsm/shakkelha)) in the ONNX format.

```python

# %% unvocalized input
input_text = "اللغة العربية هي أكثر اللغات السامية تحدثا، وإحدى أكثر اللغات انتشارا في العالم، يتحدثها أكثر من 467 مليون نسمة"

# %% shakkala output
print(vocalize(input_text, model='shakkala'))
>> اللُّغَةُ الْعَرَبِيَّةُ هِيَ أَكْثَرُ اللُّغَاتِ السَّامِيَةِ تَحَدُّثًا، وَإِحْدَى أَكْثَرِ اللُّغَاتِ انْتِشَارًا فِي الْعَالِمِ، يَتَحَدَّثُهَا أَكْثَرُ مَنْ 467 مُلْيُونُ نُسْمَةَ

# %% shakkelha output
print(vocalize(input_text, model='shakkelha'))
>> اللُّغَةُ الْعَرَبِيَّةُ هِيَ أَكْثَرُ اللُّغَاتِ السَّامِيَةِ تَحَدُّثًا، وَإِحْدَى أَكْثَرِ اللُّغَاتِ انْتِشَارًا فِي الْعَالِمِ، يَتَحَدَّثُهَا أَكْثَرُ مِنْ 467 مَلْيُونٍ نَسَمَةً


```
