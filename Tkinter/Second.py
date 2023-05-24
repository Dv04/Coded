import tkinter as tk

# Create a window
window = tk.Tk()

# Create a label
label = tk.Label(text="Hello, world!", font=("Arial", 20))

# Set the label's background color
label.config(background="#ff000f")

# Set the label's foreground color
label.config(foreground="#0ffff0")

# Add the label to the window
label.pack()

# Start the mainloop
window.mainloop()
