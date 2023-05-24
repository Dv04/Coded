import tkinter as tk

# Create the root window
root = tk.Tk()

# Create a label
label = tk.Label(root, text="Hello, world!")

# Create a button
button = tk.Button(root, text="Click me!")

# Add the label and button to the root window
label.pack()
button.pack()

# Start the main loop
root.mainloop()
