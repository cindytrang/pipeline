import sys
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from flair.data import Sentence, Corpus
from flair.datasets import ColumnCorpus
from flair.datasets import CSVClassificationCorpus
from sklearn.utils import shuffle

# for oversampling
# from imblearn import over_sampling
# from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from textblob import TextBlob
from transformers import pipeline
from sklearn.utils import class_weight


csv.field_size_limit(sys.maxsize)

# for the inappropriate content classification
pipe = pipeline("text-classification", model="michellejieli/inappropriate_text_classifier")

def create_weight_dict(filepath, delimiter, label_index):
    training_dataset = pd.read_csv(filepath, delimiter=delimiter)

    unique_labels = np.unique(training_dataset.iloc[:, label_index])
    labels = training_dataset.iloc[:, label_index]
    print("unique labels")
    print(unique_labels)

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=labels
    )
    class_weights_dict = {label: weight for label, weight in zip(unique_labels, class_weights)}

    return class_weights_dict

def add_sentiment_scores(df):
    df['sentiment'] = df['segment'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return df

def label_as_predator(row, keyword_list):
    segment = str(row['segment']) if not pd.isnull(row['segment']) else ""
    if any(word in segment.lower() for word in keyword_list):
        return "predator"
    else:
        return row['label']

def generate_variations_with_suffixes(word):
    variations = set()
    suffixes = ['', 'ing', 'ed', 's']  # TODO: Add other common suffixes 
    base_variations = generate_variations(word)
    for base_variation in base_variations:
        for suffix in suffixes:
            suffixed_variation = base_variation + suffix
            variations.update(generate_variations(suffixed_variation))
    return variations

def generate_variations(word):
    variations = {word}
    # Generate elongated variations
    for i in range(1, len(word)):
        if word[i] == word[i-1]:
            variations.update([word[:i] + word[i]*j + word[i+1:] for j in range(2, 5)])
    # Substitute common letter variations with symbols
    substitutions = {
        'a': ['@', '4'],
        'i': ['1', '!'],
        'o': ['0'],
        'e': ['3'],
        's': ['$', '5'],
    }
    for char, subs in substitutions.items():
        for sub in subs:
            if char in word:
                variations.add(word.replace(char, sub))
    variations.add(' '.join(list(word)))
    return variations

# For generating variations including suffixes
def generate_variations_for_list_with_suffixes(word_list):
    all_variations = set()
    for word in word_list:
        variations = generate_variations_with_suffixes(word)
        all_variations.update(variations)
    return all_variations

def classify_text_with_pipeline(text, pipe,  max_seq_len=510):
    if pd.isnull(text):
        return 'non-predator'
    text = text[:max_seq_len]
    try:
        result = pipe(text)
        label = result[0]['label']
        score = result[0]['score']
        return 'predator' if label == 'NSFW' and score >= 0.5 else 'non-predator'
    except Exception as e:
        print(f"Error processing text: {text}. Error: {e}")
        return 'non-predator'
    
def oversample_dataset(df, label_col: str='label'):
    X = np.array([
        df[col_name]
        for col_name in df.column_names
    ]).T

    y = df[label_col]

    sm = SMOTE(random_state=42)
    X_balanced, _ = sm.fit_resample(X, y)

    return df.Dataset.from_dict( {
            col_name: X_balanced.T[i]
            for i, col_name in enumerate(df.column_names)
        })

def undersample_based_on_min_class(df, majority_class):
    majority_df = df[df['label'] == majority_class]
    minority_df = df[df['label'] != majority_class]

    print(f"Total number of samples: {len(df)}")
    print(f"Number of majority class samples: {len(majority_df)}")
    print(f"Number of minority class samples: {len(minority_df)}")
    
    min_class_size = min(len(majority_df), len(minority_df))

    majority_sampled_df = majority_df.sample(n=min_class_size, random_state=42)
    
    undersampled_df = pd.concat([minority_df, majority_sampled_df])
    undersampled_df = shuffle(undersampled_df, random_state=42).reset_index(drop=True)
    
    return undersampled_df

def plot_2d_space(X, y, title='Classes', file_name='plot.png'):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())  # Converting sparse matrix to dense for PCA
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=150)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(scatter)
    plt.grid(True)
    plt.savefig(f'/dcs/large/u2163087/pipeline/resources/plots/{file_name}', format='png')
    plt.show()

# sexting phrases and adult related words
sexual_content_words = ['2g1c', '2 girls 1 cup', 'acrotomophilia', 'alabama hot pocket', 'alaskan pipeline', 'anal', 'anilingus', 'anus', 'apeshit', 'assmunch', 'auto erotic', 'autoerotic', 'babeland', 'baby batter', 'baby juice', 'ball gag', 'ball gravy', 'ball kicking', 'ball licking', 'ball sack', 'ball sucking', 'bangbros', 'bareback', 'barely legal', 'barenaked', 'bastard', 'bastardo', 'bastinado', 'bbw', 'bdsm', 'beaner', 'beaners', 'beaver cleaver', 'beaver lips', 'bestiality', 'big black', 'big breasts', 'big knockers', 'black cock', 'blonde action', 'blonde on blonde action', 'blowjob', 'blow job', 'blow your load', 'blue waffle', 'blumpkin', 'bollocks', 'bondage', 'boner', 'boob', 'boobs', 'booty call', 'brown showers', 'brunette action', 'bukkake', 'bulldyke', 'bullet vibe', 'bullshit', 'bung hole', 'bunghole', 'busty', 'buttcheeks', 'butthole', 'camel toe', 'camgirl', 'camslut', 'camwhore', 'carpet muncher', 'carpetmuncher', 'chocolate rosebuds', 'circlejerk', 'cleveland steamer', 'clit', 'clitoris', 'clover clamps', 'clusterfuck', 'cock', 'cocks', 'coprolagnia', 'coprophilia', 'cornhole', 'coon', 'coons', 'creampie', 'cum', 'cumming', 'cunnilingus', 'cunt', 'darkie', 'date rape', 'daterape', 'deep throat', 'deepthroat', 'dendrophilia', 'dick', 'dildo', 'dingleberry', 'dingleberries', 'dirty pillows', 'dirty sanchez', 'doggie style', 'doggiestyle', 'doggy style', 'doggystyle', 'dog style', 'dolcett', 'domination', 'dominatrix', 'dommes', 'donkey punch', 'double dong', 'double penetration', 'dp action', 'dry hump', 'dvda', 'eat my ass', 'ecchi', 'ejaculation', 'erotic', 'erotism', 'escort', 'eunuch', 'faggot', 'fecal', 'felch', 'fellatio', 'feltch', 'female squirting', 'femdom', 'figging', 'fingerbang', 'fingering', 'fisting', 'foot fetish', 'footjob', 'frotting', 'fuck buttons', 'fuckin', 'fucking', 'fucktards', 'fudge packer', 'fudgepacker', 'futanari', 'gang bang', 'gay sex', 'genitals', 'giant cock', 'girl on', 'girl on top', 'girls gone wild', 'goatcx', 'goatse', 'god damn', 'gokkun', 'golden shower', 'goodpoop', 'goo girl', 'goregasm', 'grope', 'group sex', 'g-spot', 'guro', 'hand job', 'handjob', 'hard core', 'hardcore', 'hentai', 'homoerotic', 'honkey', 'hooker', 'hot carl', 'hot chick', 'how to kill', 'how to murder', 'huge fat', 'humping', 'incest', 'intercourse', 'jack off', 'jail bait', 'jailbait', 'jelly donut', 'jerk off', 'jigaboo', 'jiggaboo', 'jiggerboo', 'jizz', 'juggs', 'kike', 'kinbaku', 'kinkster', 'kinky', 'knobbing', 'leather restraint', 'leather straight jacket', 'lemon party', 'lolita', 'lovemaking', 'make me come', 'male squirting', 'masturbate', 'menage a trois', 'milf', 'missionary position', 'motherfucker', 'mound of venus', 'mr hands', 'muff diver', 'muffdiving', 'nambla', 'nawashi', 'negro', 'neonazi', 'nigga', 'nigger', 'nig nog', 'nimphomania', 'nsfw images', 'nympho', 'nymphomania', 'octopussy', 'omorashi', 'one cup two girls', 'one guy one jar', 'orgasm', 'orgy', 'paedophile', 'paki', 'panties', 'panty', 'pedobear', 'pedophile', 'pegging', 'penis', 'phone sex', 'piece of shit', 'pissing', 'piss pig', 'pisspig', 'playboy', 'pleasure chest', 'pole smoker', 'ponyplay', 'poof', 'poon', 'poontang', 'punany', 'poop chute', 'poopchute', 'prince albert piercing', 'pthc', 'pubes', 'queaf', 'queef', 'quim', 'raghead', 'raging boner', 'rape', 'raping', 'rapist', 'rectum', 'reverse cowgirl', 'rimjob', 'rimming', 'rosy palm', 'rosy palm and her 5 sisters', 'rusty trombone', 'sadism', 'santorum', 'scat', 'schlong', 'scissoring', 'semen', 'sex', 'sexo', 'sexy', 'shaved beaver', 'shaved pussy', 'shemale', 'shibari', 'shit', 'shitblimp', 'shitty', 'shota', 'shrimping', 'skeet', 'slanteye', 'slut', 's&m', 'smut', 'snatch', 'snowballing', 'sodomize', 'sodomy', 'spic', 'splooge', 'splooge moose', 'spooge', 'spread legs', 'spunk', 'strap on', 'strapon', 'strappado', 'strip club', 'style doggy', 'suck', 'sucks', 'suicide girls', 'sultry women', 'swastika', 'swinger', 'tainted love', 'taste my', 'tea bagging', 'threesome', 'throating', 'tied up', 'tight white', 'tits', 'titties', 'titty', 'tongue in a', 'topless', 'tosser', 'towelhead', 'tranny', 'tribadism', 'tub girl', 'tubgirl', 'tushy', 'twat', 'twink', 'twinkie', 'two girls one cup', 'undressing', 'upskirt', 'urethra play', 'urophilia', 'vagina', 'venus mound', 'vibrator', 'violet wand', 'vorarephilia', 'voyeur', 'vulva', 'wank', 'wetback', 'wet dream', 'white power', 'wrapping men', 'wrinkled starfish', 'xx', 'xxx', 'yaoi', 'yellow showers', 'yiffy', 'zoophilia']# Load the dataset
all_variations = generate_variations_for_list_with_suffixes(sexual_content_words)

train_df = pd.read_csv('/dcs/large/u2163087/pipeline/datasets/Corpus/PAN12-train.csv')
test_df = pd.read_csv('/dcs/large/u2163087/pipeline/datasets/Corpus/PAN12-test.csv')

# Apply the labeling functions
train_df['label'] = train_df.apply(lambda row: label_as_predator(row, sexual_content_words), axis=1)
test_df['label'] = test_df.apply(lambda row: label_as_predator(row, sexual_content_words), axis=1)

train_df['label'] = train_df['segment'].apply(lambda x: classify_text_with_pipeline(x, pipe))
test_df['label'] = train_df['segment'].apply(lambda x: classify_text_with_pipeline(x, pipe))

#~~~~~~~~~~~~~~~~~~~~~~~~~~ Oversampling techniques ~~~~~~~~~~~~~~~~~~~~~~~~~~ 
# Initialize the vectorizer and the label encoder
# vectorizer = TfidfVectorizer(max_features=10000)
# label_encoder = LabelEncoder()

# # Drop rows where 'segment' is NaN in both train and test dataframes
# train_df = train_df.dropna(subset=['segment'])
# test_df = test_df.dropna(subset=['segment'])

# # Fill NaN values in 'label' column for both train and test dataframes
# train_df = train_df[train_df['label'].notna()]
# test_df = test_df[test_df['label'].notna()]

# label_encoder.fit(train_df['label'])

# # Transform the 'label' column in both train and test dataframes
# y_train = label_encoder.transform(train_df['label'])
# y_test = label_encoder.transform(test_df['label'])

# # Print out the mapping of original labels to encoded values
# print("Label encoding mapping:")
# for original_label, encoded_label in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
#     print(f"{original_label}: {encoded_label}")

# # Transform the 'segment' text to features using TF-IDF
# X_train = vectorizer.fit_transform(train_df['segment'])
# X_test = vectorizer.transform(test_df['segment'])

# # Visualize original data distribution
# plot_2d_space(X_train, y_train, 'Original Data Distribution', 'original_data_distribution.png')
              
# # Applying SMOTE to balance the dataset
# smote = SMOTE(random_state=42)

# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# # Visualize data distribution after SMOTE
# plot_2d_space(X_train_resampled, y_train_resampled, 'Data Distribution After SMOTE', 'after_data_distribution.png')

# # Combine and shuffle resampled data
# indices = np.arange(X_train_resampled.shape[0])
# # np.random.shuffle(indices)

# X_train_resampled = X_train_resampled[indices]
# y_train_resampled = y_train_resampled[indices]

# # Inverse transform the resampled labels back to original string labels
# y_train_resampled_str = label_encoder.inverse_transform(y_train_resampled)

# # Convert resampled data back to DataFrame for compatibility with Flair (if necessary)
# train_resampled_df = pd.DataFrame(X_train_resampled.todense(), columns=vectorizer.get_feature_names_out())
# train_resampled_df['label'] = y_train_resampled_str  

# print(train_resampled_df['label'][0])
# print("hello")
# print(pd.DataFrame(y_train_resampled_str))

#~~~~~~~~~~~~~~~~~~~~~~~~~~ Duplicate samples from the minority class to oversample ~~~~~~~~~~~~~~~~~~~~~~~~~~ 
# minority_class = 'predator'
# minority_data = train_df[train_df['label'] == minority_class]

# # Calculate how many samples you need to duplicate to balance the classes
# num_to_duplicate = train_df['label'].value_counts().max() - len(minority_data)

# # Sample from the minority data with replacement
# new_samples = minority_data.sample(n=num_to_duplicate, replace=True, random_state=42)

# # Concatenate the new samples to the original training data
# balanced_train_df = pd.concat([train_df, new_samples])

#~~~~~~~~~~~~~~~~~~~~~~~~~~ Undersample samples from the majority class ~~~~~~~~~~~~~~~~~~~~~~~~~~ 
majority_class_label = 'non-predator'

balanced_train_df = undersample_based_on_min_class(train_df, majority_class_label)

# Write to CSV files
balanced_train_df.to_csv('/dcs/large/u2163087/pipeline/datasets/Corpus/balanced_training_data.csv', index=False)
test_df.to_csv('/dcs/large/u2163087/pipeline/datasets/Corpus/testing_data.csv', index=False)

def get_corpus(data_dir, dataset_indicator):

    # column format indicating which columns hold the text and label(s)
    # column_name_map = {0: "label", 1: "chatName", 2: "segment"}
    column_name_map = {0: "label", 2: "text"}

    """
    CSVClassificationCorpus:
    Instantiates a Corpus for text classification from CSV column formatted data
    :param data_folder: base folder with the task data
    :param column_name_map: a column name map that indicates which column is text and which the label(s)
    :param train_file: the name of the train file
    :param test_file: the name of the test file
    :param dev_file: the name of the dev file, if None, dev data is sampled from train
    :param max_tokens_per_doc: If set, truncates each Sentence to a maximum number of Tokens
    :param max_chars_per_doc: If set, truncates each Sentence to a maximum number of chars
    :param use_tokenizer: If True, tokenizes the dataset, otherwise uses whitespace tokenization
    :param in_memory: If True, keeps dataset as Sentences in memory, otherwise only keeps strings
    :param fmtparams: additional parameters for the CSV file reader
    :return: a Corpus with annotated train, dev and test data
    """

    label_type = 'class'

    corpus: Corpus = CSVClassificationCorpus(
        data_dir,
        column_name_map,
        # test_file='%s-test.csv' % dataset_indicator,
        # train_file='%s-train.csv' % dataset_indicator,
        train_file='balanced_training_data.csv',
        test_file='testing_data.csv',
        label_type=label_type,
        # use_tokenizer= True,
        # delimiter = ' ',
        in_memory = False
    )

    corpus.filter_empty_sentences()
    corpus = corpus.downsample(1)
    stats = corpus.obtain_statistics()

    print("Corpus Statistics:", stats)

    return corpus