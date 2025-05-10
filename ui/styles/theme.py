import tkinter as tk
from tkinter import ttk

def configure_styles():
    style = ttk.Style()
    
    # Configure the main theme
    style.theme_use('clam')  # Use clam theme as base
    
    # Configure colors for dark theme
    style.configure('.',
        background='#2b2b2b',
        foreground='#ffffff',
        font=('Helvetica', 10)
    )
    
    # Configure frames
    style.configure('TLabelframe',
        background='#333333',
        borderwidth=2,
        relief='solid'
    )
    style.configure('TLabelframe.Label',
        background='#333333',
        foreground='#ffffff',
        font=('Helvetica', 10, 'bold')
    )
    
    # Configure buttons
    style.configure('TButton',
        padding=5,
        font=('Helvetica', 10),
        background='#4a90e2',
        foreground='#ffffff'
    )
    style.map('TButton',
        background=[('active', '#357abd'), ('disabled', '#666666')],
        foreground=[('disabled', '#999999')]
    )
    
    # Configure entry fields
    style.configure('TEntry',
        padding=5,
        fieldbackground='#404040',
        foreground='#ffffff',
        borderwidth=1
    )
    
    # Configure notebook tabs
    style.configure('TNotebook',
        background='#2b2b2b',
        borderwidth=0
    )
    style.configure('TNotebook.Tab',
        padding=[10, 5],
        font=('Helvetica', 10),
        background='#404040',
        foreground='#ffffff'
    )
    style.map('TNotebook.Tab',
        background=[('selected', '#4a90e2')],
        foreground=[('selected', '#ffffff')]
    )
    
    # Configure text widgets
    style.configure('TText',
        background='#404040',
        foreground='#ffffff',
        fieldbackground='#404040'
    )
    
    # Configure listbox
    style.configure('TListbox',
        background='#404040',
        foreground='#ffffff',
        fieldbackground='#404040'
    )
    
    # Configure scrollbar
    style.configure('TScrollbar',
        background='#404040',
        troughcolor='#2b2b2b',
        borderwidth=0
    ) 