import tkinter as tk
from src.converter import SVGPyTorchGUI


def main():
    root = tk.Tk()

    # Set macOS-specific settings
    try:
        from tkinter import ttk
        style = ttk.Style()
        style.theme_use('aqua')  # macOS native theme
    except:
        pass

    app = SVGPyTorchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()