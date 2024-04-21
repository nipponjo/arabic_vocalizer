# %%
from arabic_vocalizer import vocalize

# %% unvocalized input
input_text = "اللغة العربية هي أكثر اللغات السامية تحدثا، وإحدى أكثر اللغات انتشارا في العالم، يتحدثها أكثر من 467 مليون نسمة"

# %% shakkala output
print(vocalize(input_text, model='shakkala'))

# %% shakkelha output
print(vocalize(input_text, model='shakkelha'))

# %%