from __future__ import annotations

"""
jvlee_LIBS_ML > utils > speak.py

Function definitions to generate speech. Mainly used as alerts for when processes
finish in get_libs.py or one of the models.

"""
# region Imports
# region plain
import os 
import platform
import subprocess
# endregion
# endregion

print('speak.py loading...')

def gen_speak(text: str) -> None:
    if platform.system() == "Windows":
        # Build the powershell snippet safely
        powershell_cmd = (
            f"Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"
        )
        
        # Use subprocess.run with a list of arguments and shell=True for shell built-ins
        subprocess.run(["powershell", "-Command", powershell_cmd], shell=True)
    else:
        # Prevents execution errors on the Linux cluster environment
        print(f"🔊 Speech simulation: {text}")

if __name__ == '__main__':
    gen_speak("Hello, this is a test.")
