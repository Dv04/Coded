#!/bin/bash

# Function to check if a given year is a leap year or not
is_leap_year() {
  local year=$1
  ((year % 4 == 0 && (year % 100 != 0 || year % 400 == 0))) && return 0 || return 1
}

# Function to validate if the date is valid
is_valid_date() {
  local date_input=$1
  local day=$(echo "$date_input" | cut -d'-' -f1)
  local month=$(echo "$date_input" | cut -d'-' -f2)
  local year=$(echo "$date_input" | cut -d'-' -f3)

  if [[ $year =~ ^[0-9]{4}$ && $month =~ ^[0-9]{2}$ && $day =~ ^[0-9]{2}$ ]]; then
    if ((10#$month >= 1 && 10#$month <= 12)); then
      case $month in
        01|03|05|07|08|10|12)
          ((10#$day >= 1 && 10#$day <= 31)) && return 0 || return 1
          ;;
        04|06|09|11)
          ((10#$day >= 1 && 10#$day <= 30)) && return 0 || return 1
          ;;
        02)
          if is_leap_year "$year"; then
            ((10#$day >= 1 && 10#$day <= 29)) && return 0 || return 1
          else
            ((10#$day >= 1 && 10#$day <= 28)) && return 0 || return 1
          fi
          ;;
      esac
    fi
  fi

  return 1
}

# Read user input
echo "Enter a date (format: dd-mm-yyyy):"
read -r user_date

# Validate the entered date
if is_valid_date "$user_date"; then
  echo "The entered date ($user_date) is valid."
else
  echo "The entered date ($user_date) is not valid."
fi
