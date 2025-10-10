import json
import os

# Path to the folder containing the JSON file
folder_path = '/extracted_data/'

# Load the data from the provided JSON file inside the folder
file_path = os.path.join(folder_path, 'crime_search_results.json')

# Load the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Function to search for a keyword within the JSON data
def search_cases(keyword):
    results = {}
    for category, cases in data.items():
        matched_cases = [case for case in cases if keyword.lower() in str(case).lower()]
        if matched_cases:
            results[category] = matched_cases
    return results

# Function to display the results in a readable format
def display_case_details(results):
    if not results:
        print("No cases found for the given keyword.")
        return

    for category, cases in results.items():
        print(f"Category: {category}\n")
        for case in cases:
            print(f"Case ID: {case.get('full_case_id', 'N/A')}")
            print(f"Case Number: {case.get('case_number', 'N/A')}")
            print(f"Case Year: {case.get('case_year', 'N/A')}")
            print(f"Petitioner: {case.get('petitioner_name', 'N/A')}")
            print(f"Respondent: {case.get('respondent_name', 'N/A')}")
            print(f"Judgment Date: {case.get('judgment_date', 'N/A')}")
            print(f"Judgment Outcome: {case.get('judgment_outcome', 'N/A')}")
            print(f"Judgment Summary: {case.get('judgment_summary', 'N/A')}")
            print(f"Court: {case.get('court_name', 'N/A')}")
            print("=" * 50)

# Prompt user for search query
keyword = input("Enter a keyword to search (e.g., 'Rape'): ")
results = search_cases(keyword)

# Display results
display_case_details(results)
