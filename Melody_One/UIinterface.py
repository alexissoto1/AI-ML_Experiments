import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import messagebox
from tkinter import *

import pygame

import Model

pygame.mixer.init(44100, -16,2,2048) #Init pygame on stereo 

class Generator(object):

    def __init__(self,root):
        self.root = root

        root.title('Melody One')
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", False)

        self.frame_header = ttk.Frame(root)
        self.frame_header.pack()
        
        self.style = ttk.Style()
        self.style.configure('Header.TLabel', font = ('Arial', 18, 'bold')) 

        ttk.Label(self.frame_header, text = 'Project by Berker, Brandon and Alexis', style = 'Header.TLabel').grid(row = 0, columnspan = 1)

        self.frame_content = ttk.Frame(root)
        self.frame_content.pack()
                
        self.logo = PhotoImage(file = 'Sound.png')
        ttk.Label(self.frame_content, image = self.logo).grid(row = 5, column = 1)
        
     #Specific notes buttons.

        ttk.Button(self.frame_content, text = 'Select path', command = self.Select_Path).grid(row = 1, column = 0)
        
        ttk.Button(self.frame_content, text = 'Run', command = self.Run).grid(row = 1, column = 1)
        
        ttk.Button(self.frame_content, text = 'Check Output', command = self.Output).grid(row = 1, column = 2)
        
        ttk.Button(self.frame_content, text = 'Info', command = self.Info).grid(row = 2, column = 0)

        ttk.Button(self.frame_content, text = 'Quit', command = self._quit).grid(row = 2, column = 2)


    def Select_Path(self):
        self.path = filedialog.askdirectory() 
        print(self.path)
        return(self.path)

    def Run(self):
        Model.Run(self.path)
        messagebox.showinfo(title = 'Groove One', message = 'Done! You are ready to see the output files!')
        
    def Info(self):
        messagebox.showinfo(title = 'Groove One', message = 'Welcome! To use the app, follow the steps:' 
                     + '\n' + '\n1. Select a folder that has the MIDI files to train the model.' 
                     + '\n2. Click on "Run" to train the model and create music!'
                     + '\n3. Click on "Check Output" to play the output MIDI files created by the model.')
        
    def Output(self):

        path = filedialog.askopenfilename()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        
        
    def _quit(self):
        self.root.quit()
        self.root.destroy()
        exit()
        

def main():

    root = tk.Tk()
    app = Generator(root)
    root.mainloop()

if __name__ == "__main__": main()
