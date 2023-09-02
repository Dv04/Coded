# Below would be the complete list and use of date and time functions in python.

# Python has a module named datetime to work with dates and times.

# Import the datetime module and display the current date:

import datetime
x = datetime.datetime.now()
print(x)

# The date contains year, month, day, hour, minute, second, and microsecond.
# The datetime module has many methods to return information about the date object.
# Here are a few examples, you will learn more about them later in this chapter:

# Return the year and name of weekday:

print(x.year)
print(x.strftime("%A"))

# Creating Date Objects
# To create a date, we can use the datetime() class (constructor) of the datetime module.
# The datetime() class requires three parameters to create a date: year, month, day.

x1 = datetime.datetime(2020, 5, 17)
print(x)

# The datetime() class also takes parameters for time and timezone (hour, minute, second, microsecond, tzone), but they are optional, and has a default value of 0, (None for timezone).

# The strftime() Method
# The datetime object has a method for formatting date objects into readable strings.

# The method is called strftime(), and takes one parameter, format, to specify the format of the returned string:

# Display the name of the month:

print(x.strftime("%B"))

# A reference of all the legal format codes:
# Here is an example of using all of the formats.

x = datetime.datetime.now()

print("The following formats are available:")
print("a = Weekday, short version: ", x.strftime("%a"))
print("A = Weekday, full version: ", x.strftime("%A"))
print("w = Weekday as a number 0-6, 0 is Sunday: ", x.strftime("%w"))
print("d = Day of month 01-31: ", x.strftime("%d"))
print("b = Month name, short version: ", x.strftime("%b"))
print("B = Month name, full version: ", x.strftime("%B"))
print("m = Month as a number 01-12: ", x.strftime("%m"))
print("y = Year, short version, without century: ", x.strftime("%y"))
print("Y = Year, full version: ", x.strftime("%Y"))
print("H = Hour 00-23: ", x.strftime("%H"))
print("I = Hour 00-12: ", x.strftime("%I"))
print("p = AM/PM: ", x.strftime("%p"))
print("M = Minute 00-59: ", x.strftime("%M"))
print("S = Second 00-59: ", x.strftime("%S"))
print("f = Microsecond 000000-999999: ", x.strftime("%f"))
print("z = UTC offset: ", x.strftime("%z"))
print("Z = Timezone: ", x.strftime("%Z"))
print("j = Day number of year 001-366: ", x.strftime("%j"))
print("U = Week number of year, Sunday as the first day of week, 00-53: ", x.strftime("%U"))
print("W = Week number of year, Monday as the first day of week, 00-53: ", x.strftime("%W"))
print("c = Local version of date and time: ", x.strftime("%c"))
print("x = Local version of date: ", x.strftime("%x"))
print("X = Local version of time: ", x.strftime("%X"))
print("%% = A % character: ", x.strftime("%%"))

# The strptime() Method
# The datetime object has a method for creating a datetime object from a string.

# The method is called strptime(), and takes two parameters, format and string.

# The format parameter defines the format that the date string is written in.

# The datetime.strptime() method returns a datetime object.

# Create a datetime object from a string:

x = datetime.datetime.strptime("17 May 2020", "%d %B %Y")
print(x)


