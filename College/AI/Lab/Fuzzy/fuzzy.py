import sys
from fuzzywuzzy import fuzz
from itertools import combinations, permutations
import csv
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import argparse

def string_similarity(str1, str2, method='ratio'):
    if method == 'ratio':
        # standard Levenshtein distance similarity ratio
        similarity = fuzz.ratio(str1, str2)
    elif method == 'partial_ratio':
        # partial ratio (good for partial matching)
        similarity = fuzz.partial_ratio(str1, str2)
    elif method == 'token_sort_ratio':
        # ignoring word order ('fuzzy was a bear' vs 'was fuzzy a bear?')
        similarity = fuzz.token_sort_ratio(str1, str2)
    else:
        raise ValueError(f"Unknown method: {method}")

    formatted_similarity = round(similarity, 1)  # Round to 1 decimal place
    return formatted_similarity

def get_comparison_method():
    print("\nPlease select the comparison method:")
    print("1: Simple Ratio (ratio)")
    print("2: Partial Ratio (partial_ratio)")
    print("3: Token Sort Ratio (token_sort_ratio)")
    print("4: All Methods")
    selection = input("Enter the number of the method or method name: ")

    method = selection.lower()
    if method in ['1', 'ratio', 'simple ratio']:
        return ['ratio']
    elif method in ['2', 'partial_ratio', 'partial ratio']:
        return ['partial_ratio']
    elif method in ['3', 'token_sort_ratio', 'token sort ratio']:
        return ['token_sort_ratio']
    elif method in ['4', 'all', 'all methods']:
        return ['ratio', 'partial_ratio', 'token_sort_ratio']
    else:
        print("Invalid selection. Defaulting to 'Simple Ratio'")
        return ['ratio']

def input_strings():
    strings = []
    print("\nPlease enter any number of strings you want. Just press enter on an empty line when you're done.\n")
    while True:
        try:
            inp = input("Enter a string (or press enter to finish): ")
            if inp == "":
                if strings:  # Check if there's at least one string entered
                    break
                else:
                    print("You must enter at least one string. Please try again.")
            else:
                strings.append(inp)
        except KeyboardInterrupt:
            print("\nInput interrupted by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
    return strings

def parallel_comparison(pairs, method):
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as per your system's specifications
        future_to_pair = {executor.submit(string_similarity, pair[0], pair[1], method): pair for pair in pairs}
        for future in concurrent.futures.as_completed(future_to_pair):
            pair = future_to_pair[future]
            try:
                results[pair] = future.result()
            except Exception as e:
                print(f"Generated an exception: {e}")
    return results

def compare_strings(strings, methods, comparison_type):
    all_similarities = {}
    pairs = combinations(strings, 2)  # or permutations, based on 'comparison_type'
    
    for method in methods:
        print(f"\nProcessing {method} method...")
        results = parallel_comparison(pairs, method)
        all_similarities[method] = results

    # Display similarities and save the report
    for method, similarities in all_similarities.items():
        print(f"\nResults using {method} method:")
        for pair, similarity in similarities.items():
            print(f"Similarity between '{pair[0]}' and '{pair[1]}' is {similarity}%")

        # Consider saving to a different file for each method or adjust the structure of your CSV
        save_to_csv(similarities, method, filename=f'comparison_report_{method}.csv')

    return all_similarities

def save_to_csv(data, method, filename=None):
    # If no filename is provided, generate one based on the method
    if filename is None:
        filename = f'comparison_report_{method}.csv'
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Adjust the header to include the method
        writer.writerow(['Method', 'String 1', 'String 2', 'Similarity'])
        
        for pair, similarity in data.items():
            # Write the method along with the other data
            writer.writerow([method, pair[0], pair[1], f'{similarity}%'])

    print(f"\nReport saved to {filename}")

def main_menu():
    while True:
        print("\nString Comparison Tool")
        print("1: Input strings and compare")
        print("2: Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            # Input strings from user
            strings = input_strings()

            # Get comparison method and type from user
            methods = get_comparison_method()
            comparison_type = 'pairs'

            # Compare strings and get similarities
            all_similarities = compare_strings(strings, methods, comparison_type)

        elif choice == '2':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please select a valid option.")

if __name__ == "__main__":

    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")
        sys.exit(0)
