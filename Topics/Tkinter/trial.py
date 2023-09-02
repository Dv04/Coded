import tkinter as tk

# Create the root window
root = tk.Tk()

# Set the background image
root.configure(background="lightblue")

# Create a label
label = tk.Label(root, text="Hello, world!", font=("Times New Roman", 20),background="lightblue")
label1 = tk.Label(root, text="Hello, world!", font=("Arial", 20),background="lightblue")

# Create a button
button = tk.Button(root, text="Click me!", font=("Arial", 15),background="lightblue")

# Add the label and button to the root window
label.pack()
label1.pack()
button.pack()

# Start the main loop
root.mainloop()
