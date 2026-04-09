import os 

print('speak.py loading...')

def gen_speak(text):
    os.system(f'powershell -Command "Add-Type -AssemblyName System.Speech; '
                f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"')
        

if __name__ == '__main__':
    gen_speak("Hello, this is a test.")
