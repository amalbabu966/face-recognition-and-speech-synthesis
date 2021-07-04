import gtts
import sys
from playsound import playsound
import pandas as pd
import translators as ts

flag = 0
try:
    parameter1 = sys.argv[1]
    # parameter2 = sys.argv[2]
except:
    flag = 1
    teacher_text = "This teacher is not exist in our records hehe"

if flag == 0:
    information = {
        "arun": "Hi , His name is Arun B I. He is an assistant professor at University College of engineering karyavattam.He  is one of the best Teachers that we have",
        "meharu": "Hi ,This is Meharuniza nazzem . Assistant professor at University  College of engineering karyavattam.she is one of the pre-eminent trs that we have",
        "sabeena": "Hi , She is sabeena A .S. she is one of the assistant  professor at University College of engineering karyavattam.She is one of the finest teachers that we have.",
        "reshma": "Reshma R, she is one of the assistant professor at university college of engineering kariavatton. She is the staff advisor of final year Information technology."
    }
    try:
        teacher_text = information[parameter1]
    except:
        teacher_text = "Sorry! I can't find details about this person."


translate_text = ts.google(teacher_text,from_language='en',to_language='ml')
# translate_text = ts.bing(teacher_text,from_language='en',to_language='ml')

# print(translate_text)


text_val = translate_text
language = 'ml'

t1 = gtts.gTTS(text=text_val, lang=language, slow=False)
t1.save("welcome.mp3")
playsound("welcome.mp3")
