import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Load dataset
df = pd.read_csv('support_tickets.csv')

# 1. Count tickets by category
category_counts = df['Category'].value_counts()
print("Tickets by Category:")
print(category_counts)

# 2. Extract and sort resolution times
df['ResolutionHours'] = df['ResolutionTime'].str.extract(r'(\d+\.?\d*)').astype(float)
resolution_summary = df[['TicketID', 'Category', 'ResolutionHours']].sort_values(by='ResolutionHours', ascending=False)
print("\nTop 5 Tickets by Resolution Time:")
print(resolution_summary.head())

# 3. Count tickets by status
status_counts = df['Status'].value_counts()
print("\nTickets by Status:")
print(status_counts)

# 4. Extract keywords from IssueDescription
nltk.download('punkt')
nltk.download('punkt_tab')
keywords = []
for desc in df['IssueDescription']:
    tokens = word_tokenize(desc.lower())
    keywords.extend([word for word in tokens if word.isalpha()])

keyword_counts = Counter(keywords)
print("\nTop 10 Keywords:")
print(keyword_counts.most_common(10))

# Save results to CSVs
category_counts.to_csv('category_counts.csv')
resolution_summary.to_csv('resolution_summary.csv', index=False)
status_counts.to_csv('status_counts.csv')
pd.DataFrame(keyword_counts.most_common(10), columns=['Keyword', 'Count']).to_csv('keyword_counts.csv', index=False)