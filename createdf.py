import pandas as pd
from langdetect import detect, DetectorFactory
import langdetect
import emoji



'''
# Reading the CSV files
df1 = pd.read_csv('tweeter.csv')
df2 = pd.read_csv('cops.csv')
df3 = pd.read_csv('Extended_Tweets_Dataset.csv')
df4 = pd.read_excel('tweets_updated_e.xlsx')

# Renaming columns
df3.rename(columns={'Tweet': 'tweets'}, inplace=True)
'''
def replace_emoji_with_text(text):
    # This function replaces emojis with their text representation
    return emoji.demojize(text)

'''
# Replacements for Action column
replacements = {'Action required': 1, 'action required': 1,         'action not required': 0}

# Replace the values in the 'Action' column for df1
df1['Action'] = df1['Action'].replace(replacements)

# Concatenating the DataFrames
final_df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
final_df = pd.concat([final_df, df4], axis=0, ignore_index=True)
'''
# Function to check if text is in English
def is_english(text):
    if isinstance(text, str):
        try:
            return detect(text) == 'en'
        except langdetect.lang_detect_exception.LangDetectException:
            return False
    return False

'''
# Apply the function to check for English tweets
final_df['is_english'] = final_df['tweets'].apply(is_english)

# Filter and clean the DataFrame
final_df = final_df[final_df['is_english']].drop(columns=['is_english'])
final_df.reset_index(drop=True, inplace=True)

final_df =pd.read_excel('concatenated_file2.xlsx')
final_df['tweets'] = final_df['tweets'].apply(replace_emoji_with_text)

# Output to Excel
#final_df.to_excel('concatenated_file3.xlsx', index=False)

print(final_df.head())
print(final_df.tail())
'''

