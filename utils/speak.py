"""
jvlee_LIBS_ML > utils > speak.py

Function definitions to generate speech. Mainly used as alerts for when processes
finish in get_libs.py or one of the models.

"""
# region Imports
# region plain
import os 
# endregion
# endregion

print('speak.py loading...')

def gen_speak(text):
    os.system(f'powershell -Command "Add-Type -AssemblyName System.Speech; '
                f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"')
        

if __name__ == '__main__':
    gen_speak("Hello, this is a test.")
