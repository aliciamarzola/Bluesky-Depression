import pandas as pd 
import numpy as np
import emoji 

scraper3 = pd.read_csv('scrapers/scraper3.csv')
scraper4 = pd.read_csv('scrapers/scraper4.csv')

queries = ['i want to disappear', 'nothing matters',
       "i'm tired of everything", "can't take it anymore",
       "i don't want to be here", 'feeling great', 'i feel empty']

scraper4 = scraper4.loc[scraper4['query'].isin(queries)]

englishlang = ['en', 'en-US', 'uk']
scraper4 = scraper4.loc[(scraper4['langs/0'].isin(englishlang))|(scraper4['langs/1'].isin(englishlang))]
scraper3 = scraper3.loc[(scraper3['langs/0'].isin(englishlang))|(scraper3['langs/1'].isin(englishlang))]


dataset = pd.concat([scraper3, scraper4])
dataset['depressive'] = 0

dataset = dataset.drop_duplicates(subset=['text'])
dataset = dataset.dropna(subset=['text'])



dataset = dataset[['text', 'depressive']]

def remove_emojis(text):
    return emoji.replace_emoji(text, replace='') if isinstance(text, str) else text

dataset['text'] = dataset['text'].apply(remove_emojis)
print(len(dataset))

dataset.to_csv(f"dataset_novo.csv", index=False)