from setuptools import setup
 
import sys
sys.setrecursionlimit(1500)  # You can adjust the number as needed

APP = ['offline_ui.py']
DATA_FILES = []
OPTIONS = {
       'argv_emulation': True,
       'packages': ['tkinter', 'cv2', 'PIL', 'torch', 'transformers'],
       'iconfile': 'icon.icns',  # Use the newly created icon file
   }

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
) 