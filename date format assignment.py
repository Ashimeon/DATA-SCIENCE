import pandas as pd

# Create a dummy dataset
data = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'country': ['USA', 'U.S.A.', 'United States', 'Canada', 'Mexico'],
    'date': ['2021-01-01', '01/01/2021', '2021-01-01', '2021-01-01', '01-01-2021']
})

# Define rules for resolving inconsistencies
country_codes = {
    'USA': 'US',
    'U.S.A.': 'US',
    'United States': 'US',
    'Canada': 'CA',
    'Mexico': 'MX'
}
print("Inconsistent Data")
print(data)

def parse_date(date_str):
    if '-' in date_str:
        parts = date_str.split('-')
        if len(parts[0]) != 4:
            year=max(parts)
            return f'{year}-{parts[1]}-{parts[0]}'
        else:
            return date_str
    elif '/' in date_str:
        parts = date_str.split('/')
        return f'{parts[2]}-{parts[0]}-{parts[1]}'

# Apply rules to the data
data['country'] = data['country'].apply(lambda x: country_codes.get(x, x))
data['date'] = data['date'].apply(parse_date)

# Verify the results
print("\nConsistent Data")
print(data)