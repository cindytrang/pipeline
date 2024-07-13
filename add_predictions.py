import setup_tensorflow
import json
import os
import re
import ast
import argparse
import spacy
import torch
import ast


from concrete_classes import getTFLiteModel
from eval_util import getEvalArgs, iterate_multi_threaded, contentToString, isNonemptyMsg, breakUpChat, get_classifier, MasterClassifier
from pathlib import Path
from datetime import datetime, time, timezone  
from transformers import pipeline
from torchvision import models, transforms
from PIL import Image


def getUNIXTimestamp():
	return int(datetime.now().replace(tzinfo=timezone.utc).timestamp()*1000)

# Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Load the sentiment analysis pipeline from Transformers
sentiment_pipeline = pipeline("sentiment-analysis")

parser = argparse.ArgumentParser(description='Evaluate a model')
parser.add_argument(
	"--eval_mode",
	dest='eval_mode',
	help="whether to evaluate with complete predator chats or (predator and non-predator) segments of chats. *_fast modes are recommended. They speeds up the respecitve mode by only analyzing until the first warning is raised.",
	choices=["segments", "segments_fast", "full", "full_fast"],
	required=False
)

parser.add_argument(
	"--window_size",
	dest='window_size',
	help="we look at the last `window_size` messages during classification",
	type=int,
	default=50
)

relate_build_trust_keywords = ['me too', 'i also', 'i feel', 'i understand', 'are you sad?', 'like an adult', ' being lonely']

isolation_keywords = ['alone', 'by yourself', 'only child', 'divorced',   "as I say", "if u", "tell you", "bitch", "if u don't", "telling u",
    "dare", "if you", "telling you", "do it", "if you don't", "told to","don't u dare", "just do it", "told u", "don't you dare", "know where u",
    "told you", "dont", "know where you", "trouble", "dont eva", "lil slut", "trubbl","dont ever", "little bitch", "u better", "dont you ever", "little slut", "u remember",
    "enjoy", "my bitch", "wet", "fault for being", "need to", "what I say", "fault that", "now", "where u live", "hard", "or I", "where you live",
	"i know where", "remember", "white slut", "slut", "tell u", "you better", 'looking for a younger' 'You', 'Friend', 'boyfriend', 'girlfriend', 'lover', 'Adult', 'anyone', 'personal', 'party', 'outsider', 
    'fight', 'story', 'mentions', 'dating', 'helpful', 'phone', 'private', 'public', 'gossip', 'Homework', 
    'office', 'school', 'Art', 'bands', 'game', 'hangout', 'sport', 'television', 'movie',  'Cares', 'casual', 'cherish', 'comfort', 'cute', 'nice', 'LMAO', 'Best', 'better', 'confidence', 
    'control', 'important', 'work', 'Income', 'store', 'value', 'rich', 'wealth', 'compensate', 
    'Church', 'God', 'heaven', 'hell', 'sacred', 'paradise'
]

personal_info_keywords = [
    'where do you live', 'how old are you', 'old are you', 'old are u','what school', 'what grade','to old for u', 'too old for u', ' the oldest age u would date', 'Daughter', 'mother', 'husband', 'aunt', 'Apartment', 
    'Crap', 'cry', 'difficult', 'hate', 'heartbreak', 'tough', 'unimportant', 'punish', 'sad', 'lose', 'kitchen', 'family', 'They', 'their', 'they’d', 
    'Worried', 'fearful', 'nervous', 'do u go to school', 'where is ur mom', 'u got a pic', ' where u frum', 'how old', 'do you like older guys?', 'the oldest u would date', 'can i ask a personal question first',
	'turned 13 last','13 m','am 13','almost 11','will be 5 next','just 7 years old','almost 7','turned 15 last','turned 13 last','almost 10','turned 6 last','13 m','im 10','im over 8','am 14','almost 6','you are 15','only 14','14 year','am 10','you are 16','im over 8','turned 17 last','9 m','just turned 16','16 f','7 m','im 5','just 14 years old','5 m','turned 9 last','turned 9 last','16 year','only 8','u are 12','im over 6','just 12 years old','just 17 years old','just turned 13','10 year','can you believe I am 5','11 m','11 f','10 f','will be 10 next','almost 8','can you believe I am 15','almost 9','am 7','just turned 10','just 13 years old','im over 17','you are 16','only 9','8 m','just 12 years old','11 year','im over 12','can you believe I am 9','turned 14 last','u are 16','almost 6','will be 8 next','10 m','13 year','u are 17','im over 14','am 14','just turned 5','11 f','am 10','17 f','will be 9 next','turned 15 last','can you believe I am 5','10 f','u are 11','almost 5','can you believe I am 8','6 m','13 year','12 m','just 11 years old','only 7','can you believe I am 12','you are 10','will be 6 next','im 17','turned 11 last','can you believe I am 10','can you believe I am 17','8 year','just turned 14','12 f','only 6','turned 10 last','only 12','just 10 years old','im 17','almost 11','just turned 7','im over 16','im 6','6 f','can you believe I am 5','will be 12 next','im 12','am 14','you are 14','am 16','will be 17 next','can you believe I am 16','just turned 5','am 6','turned 6 last','12 m','only 5','will be 9 next','just turned 17','u are 10','14 year','im over 9','just 11 years old','am 8','can you believe I am 17','just turned 8','turned 10 last','only 8','almost 16','turned 12 last','just 6 years old','you are 11','am 5','17 year','am 15','only 10','almost 11','im 10','just 13 years old','only 6','will be 8 next','will be 7 next','turned 14 last','just 12 years old','17 year','just turned 7','just 11 years old','11 f','im 13','8 year','am 11','will be 11 next','im 7','only 17','10 year','can you believe I am 13','im over 16','can you believe I am 8','can you believe I am 5','will be 5 next','im over 6','16 m','13 f','im 5','just 10 years old','almost 14','only 17','just turned 5','only 17','only 17','7 year','you are 13','im 6','u are 8','will be 15 next','only 8','12 year','turned 10 last','will be 15 next','u are 14','can you believe I am 5','u are 9','u are 11','15 f','u are 8','7 year','im over 9','you are 15','5 m','turned 10 last','you are 9','9 m','am 9','will be 12 next','only 17','almost 10','16 f','just 8 years old','will be 7 next','im over 5','will be 14 next','almost 5','just 15 years old','only 8','only 5','just turned 15','im 17','u are 6','almost 10','only 7','am 16','only 7','will be 9 next','turned 8 last','just 9 years old','can you believe I am 13','im over 16','15 year','almost 6','turned 12 last','10 f','you are 15','turned 13 last','im 13','just 14 years old','turned 11 last','12 year','just 16 years old','just 6 years old','can you believe I am 10','just turned 14','am 7','7 year','turned 12 last','can you believe I am 11','can you believe I am 17','am 7','13 m','only 16','just turned 15','15 m','10 year','almost 13','im 5','will be 11 next','17 m','only 5','17 m','u are 5','6 m','only 17','im over 5','you are 16','almost 9','just turned 10','can you believe I am 16','almost 15','can you believe I am 12','just turned 17','16 year','will be 16 next','11 f','8 year','you are 6','15 m','am 6','17 m','can you believe I am 5','am 15','will be 15 next','just turned 10','just turned 8','you are 13','turned 6 last','14 m','im 16','can you believe I am 10','11 year','just 5 years old','almost 8','10 f','turned 7 last','only 12','just turned 10','u are 9','13 year','just 14 years old','am 11','turned 5 last','im over 13','you are 6','will be 17 next','will be 10 next','im over 7','u are 6','im 13','you are 15','u are 13','only 7','just 8 years old','only 12','am 11','im 12','just turned 13','you are 12','you are 11','only 8','will be 17 next','you are 10','am 15','turned 11 last','im over 17','just turned 12','just 6 years old','you are 9','will be 7 next','am 12','14 year','turned 5 last','just turned 13','im 17','just 8 years old','10 f','you are 10','im over 15','im over 15','8 year','just 16 years old','almost 10','am 17','11 year','im over 8','6 f','u are 17','can you believe I am 13','im over 14','u are 8','16 year','15 year','turned 12 last','almost 9','you are 17','only 15','u are 6','turned 13 last','im over 13','almost 8','can you believe I am 6','7 f','im 12','you are 9','16 year','10 f','just 5 years old','im 5','will be 9 next','16 m','turned 14 last','can you believe I am 16','turned 9 last','im 5','16 f','just 15 years old','only 9','8 year','u are 12','almost 16','just 8 years old','just turned 12','im over 5','turned 7 last','almost 6','u are 13','13 f','im 11','im 16','can you believe I am 8','only 12','just turned 6','5 f','can you believe I am 17','only 15','only 12','you are 5','just 16 years old','turned 12 last','9 year','almost 6','you are 16','9 f','im over 13','6 year','im 9','you are 7','can you believe I am 13','will be 5 next','will be 5 next','am 14','will be 5 next','im over 11','6 year','16 m','7 year','im over 16','11 m','14 f','8 m','can you believe I am 7','im over 9','im over 16','u are 15','u are 6','7 f','will be 15 next','turned 17 last','you are 9','will be 7 next','just 7 years old','you are 17','only 9','can you believe I am 10','only 8','will be 6 next','im over 8','14 m','im 17','u are 9','only 10','im over 13','14 f','8 m','just 9 years old','only 9','im 8','will be 17 next','am 11','am 6','turned 14 last','you are 15','almost 5','im over 10','only 12','you are 16','u are 7','8 year','am 8','im 17','just 8 years old','15 f','im 13','im 12','am 17','15 m','can you believe I am 13','u are 15','11 m','am 7','u are 16','16 f','12 year','can you believe I am 9','will be 9 next','will be 14 next','just turned 6','you are 11','im over 11','will be 6 next','16 f','am 17','you are 15','will be 12 next','almost 14','turned 11 last','only 10','just 13 years old','turned 12 last','turned 5 last','7 m','will be 16 next','almost 14','turned 10 last','im 12','almost 12','im over 11','will be 14 next','9 f','just turned 8','only 16','am 12','16 m','you are 14','11 year','can you believe I am 16','just turned 15','you are 5','am 9','im 12','11 m','u are 13','turned 5 last'
]

sexual_content_words = ['2g1c', '2 girls 1 cup', 'acrotomophilia', 'alabama hot pocket', 'alaskan pipeline', 'anal', 'anilingus', 'anus', 'apeshit', 'arsehole', 'ass', 'asshole', 'assmunch', 'auto erotic', 'autoerotic', 'babeland', 'baby batter', 'baby juice', 'ball gag', 'ball gravy', 'ball kicking', 'ball licking', 'ball sack','nekkid', 'ball sucking', 'bangbros', 'bareback', 'barely legal', 'barenaked', 'bastard', 'bastardo', 'bastinado', 'bbw', 'bdsm', 'beaner', 'beaners', 'beaver cleaver', 'beaver lips', 'bestiality', 'big black', 'big breasts', 'big knockers', 'big tits', 'bimbos', 'birdlock', 'bitch', 'bitches', 'black cock', 'blonde action', 'blonde on blonde action', 'blowjob', 'blow job', 'blow your load', 'blue waffle', 'blumpkin', 'bollocks', 'bondage', 'boner', 'boob', 'boobs', 'booty call', 'brown showers', 'brunette action', 'bukkake', 'bulldyke', 'bullet vibe', 'bullshit', 'bung hole', 'bunghole', 'busty', 'butt', 'buttcheeks', 'butthole', 'camel toe', 'camgirl', 'camslut', 'camwhore', 'carpet muncher', 'carpetmuncher', 'chocolate rosebuds', 'circlejerk', 'cleveland steamer', 'clit', 'clitoris', 'clover clamps', 'clusterfuck', 'cock', 'cocks', 'coprolagnia', 'coprophilia', 'cornhole', 'coon', 'coons', 'creampie', 'cum', 'cumming', 'cunnilingus', 'cunt', 'darkie', 'date rape', 'daterape', 'deep throat', 'deepthroat', 'dendrophilia', 'dick', 'dildo', 'dingleberry', 'dingleberries', 'dirty pillows', 'dirty sanchez', 'doggie style', 'doggiestyle', 'doggy style', 'doggystyle', 'dog style', 'dolcett', 'domination', 'dominatrix', 'dommes', 'donkey punch',' any sexual experience' ,'double dong', 'double penetration', 'dp action', 'dry hump', 'dvda', 'eat my ass', 'ecchi', 'ejaculation', 'erotic', 'erotism', 'escort', 'eunuch', 'faggot', 'fecal', 'felch', 'fellatio', 'feltch', 'female squirting', 'femdom', 'figging', 'fingerbang', 'fingering', 'fisting', 'foot fetish', 'footjob', 'frotting', 'fuck', 'fuck buttons', 'fuckin', 'fucking', 'fucktards', 'fudge packer', 'fudgepacker', 'futanari', 'gang bang', 'gay sex', 'genitals', 'giant cock', 'girl on', 'girl on top', 'girls gone wild', 'goatcx', 'goatse', 'god damn', 'gokkun', 'golden shower', 'goodpoop', 'goo girl', 'goregasm', 'grope', 'group sex', 'g-spot', 'guro', 'hand job', 'handjob', 'hard core', 'hardcore', 'hentai', 'homoerotic', 'honkey', 'hooker', 'hot carl', 'hot chick', 'how to kill', 'how to murder', 'huge fat', 'humping', 'incest', 'intercourse', 'jack off', 'jail bait', 'jailbait', 'jelly donut', 'jerk off', 'jigaboo', 'jiggaboo', 'jiggerboo', 'jizz', 'juggs', 'kike', 'kinbaku', 'kinkster', 'kinky', 'knobbing', 'leather restraint', 'leather straight jacket', 'lemon party', 'lolita', 'lovemaking', 'make me come', 'male squirting', 'masturbate', 'menage a trois', 'milf', 'missionary position', 'motherfucker', 'mound of venus', 'mr hands', 'muff diver', 'muffdiving', 'nambla', 'nawashi', 'negro', 'neonazi', 'nigga', 'nigger', 'nig nog', 'nimphomania', 'nipple', 'nipples', 'nsfw images', 'nude', 'nudity', 'nympho', 'nymphomania', 'octopussy', 'omorashi', 'one cup two girls', 'one guy one jar', 'orgasm', 'orgy', 'paedophile', 'paki', 'panties', 'panty', 'pedobear', 'pedophile', 'pegging', 'penis', 'phone sex', 'piece of shit', 'pissing', 'piss pig', 'pisspig', 'playboy', 'pleasure chest', 'pole smoker', 'ponyplay', 'poof', 'poon', 'poontang', 'punany', 'poop chute', 'poopchute', 'porn', 'porno', 'pornography', 'prince albert piercing', 'pthc', 'pubes', 'pussy', 'queaf', 'queef', 'quim', 'raghead', 'raging boner', 'rape', 'raping', 'rapist', 'rectum', 'reverse cowgirl', 'rimjob', 'rimming', 'rosy palm', 'rosy palm and her 5 sisters', 'rusty trombone', 'sadism', 'santorum', 'scat', 'schlong', 'scissoring', 'semen', 'sex', 'sexo', 'sexy', 'shaved beaver', 'shaved pussy', 'shemale', 'shibari', 'shit', 'shitblimp', 'shitty', 'shota', 'shrimping', 'skeet', 'slanteye', 'slut', 's&m', 'smut', 'snatch', 'snowballing', 'sodomize', 'sodomy', 'spic', 'splooge', 'splooge moose', 'spooge', 'spread legs', 'spunk', 'strap on', 'strapon', 'strappado', 'strip club', 'style doggy', 'suicide girls', 'sultry women', 'swastika', 'swinger', 'tainted love', 'taste my', 'tea bagging', 'threesome', 'throating', 'tied up', 'tight white', 'tit', 'tits', 'titties', 'titty', 'tongue in a', 'topless', 'tosser', 'towelhead', 'touch yourself', 'tranny', 'tribadism', 'tub girl', 'tubgirl', 'tushy', 'twat', 'twink', 'twinkie', 'two girls one cup', 'undressing', 'upskirt', 'urethra play', 'urophilia', 'vagina', 'venus mound', 'vibrator', 'violet wand', 'vorarephilia', 'voyeur', 'vulva', 'wank', 'wetback', 'wet dream', 'white power', 'wrapping men', 'wrinkled starfish', 'xx', 'xxx', 'yaoi', 'yellow showers', 'yiffy', 'zoophilia', 'want fuck', 'please need', 'let cum', 'cum mouth', 'oh god', 'please cum', 'oh fuck', 'fuck thats', 'im going', 'think ’', '’ like', 'baby ’', 'nipples hard', 'yeah ’', '’ ’', 'index finger', 'finger pussy', 'good girl', 'middle finger', 'pussy lips', 'take time', 'kissing neck', 'hard right', '’ hot', 'hot hot', 'know ’', '’ going', 'baby oh', 'oh please', 'keep going', 'eat ass', 'gon na', 'na fuck', 'ok ’', '’ back', 'wan na', '’ believe', 'legs crossed', 'finger clit', 'big dick', 'go slow', 'hard cock', 'hard dick', 'im hard', 'hard fuck', 'lay bed', 'hard hard', 'want feel', 'want suck', 'fuck mouth', 'want lay', 'u want', 'want kiss', 'oh yeah', 'pussy tongue', 'dont care', 'im sucking', 'like crazy', 'inside dripping', 'dont get', 'baby girl', 'dick inside', 'fuck slowly', 'cum cock', 'cum face', '’ headed', 'game come', 'send picture', '’ right', 'san diego', 'na see', '’ good', '’ keep', 'belligerent xs', 'xs lol', 'spin class', 'cartoon network', 'color run', 'kitchen counter', 'send pic', 'want see', 'see dick', 'see little', 'really want', '’ laundry', 'laundry day', '’ worry', 'maybe little', 'little pussy', 'spread legs', 'wet pussy', 'going turn', 'cock deep', 'deep inside', 'want fill', 'make scream', 'big hard', 'dick tight', 'tight pussy', 'bent bed', 'want hands', 'want make', 'kiss body', 'neck pulling', 'want bad', 'rub clit', 'harder harder', 'finger finger', 'im gon', 'yeah want', 'inside thighs', 'make come', 'come mouth', 'pretty little', 'cock bad', 'god want', 'cant take', 'take anymore', 'want play', 'wont stop', 'could please', 'slowly spread', 'throat making', 'id love', 'private places', 'princess plug', 'cant wait', ]

labels = ["Gathering Information and Selecting the Victim",
          "Trust Development and Establishing Credibility",
          "Priming and Desensitising the Target"]

args = getEvalArgs(parser)

datapackPath = os.path.join(args.data_dir,'PJZ.json')
	# 'datapack-%s-test.json' % args.dataset_indicator)
with open(datapackPath, "r") as file:
	datapack = json.load(file)
 
# ALEXNET FOR IMAGE CLASSIFICATION
alexnet = models.alexnet(pretrained=True)
alexnet.eval()  
transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

#--------------------------------------- Print the structure of the datapack--------------------------------------
print("Datapack structure:")
print(json.dumps(datapack, indent=4, sort_keys=True)[:1000])
chatNames = sorted(list(datapack["chats"].keys()))

def contains_personal_info_keywords(text, keywords):
	text_lower = text.lower()
	return any(keyword in text_lower for keyword in keywords)

def contains_keywords(text, keywords):
	text_lower = text.lower()
	return any(keyword in text_lower for keyword in keywords)

def contains_sexual_content(message_body, sexual_content_words):
	pattern = '|'.join(map(re.escape, sexual_content_words))
	return re.search(pattern, message_body, re.IGNORECASE) is not None

def is_time_in_range(timestamp, start_time, end_time):
	chat_datetime = datetime.fromtimestamp(timestamp / 1000)
	chat_time = chat_datetime.time()
	if start_time <= end_time:
		return start_time <= chat_time <= end_time
	else:
		return chat_time >= start_time or chat_time <= end_time
	
def load_imagenet_classes(file_path):
	with open(file_path, 'r') as file:
		content = file.read()
		classes_dict = ast.literal_eval(content)
	return classes_dict

def load_imagenet_classes(file_path):
	with open(file_path, 'r') as file:
		content = file.read()
		classes_dict = ast.literal_eval(content)
	return classes_dict

def analyze_image(image_path, imagenet_classes):
	try:
		img = Image.open(image_path)
		img_t = transform(img)
		batch_t = torch.unsqueeze(img_t, 0)
		with torch.no_grad():
			out = alexnet(batch_t)
		_, index = torch.max(out, 1)
		index = index.item()
		label = imagenet_classes.get(index, "Unknown")
		return label, index
	except FileNotFoundError:
		print(f"Error: Image file not found at {image_path}.")
		return None, None
	except Exception as e:
		print(f"An error occurred during image analysis: {e}")
		return None, None

def normalize_score(score, min_observed=0.4, max_observed=0.65):
    normalized_score = (score - min_observed) / (max_observed - min_observed)
    return max(0, min(1, normalized_score))

if chatNames:
	firstChatName = chatNames[0]
	firstChatContent = datapack["chats"][firstChatName]
	for message in firstChatContent['content']:
		timestamp = message['time']
	firstMessage = datapack["chats"][firstChatName]["content"][0]
	firstMessage["labels"] = [labels[1]]
	print("\nSample chat segment (First chat name: {})".format(firstChatName))
	print(json.dumps(firstChatContent, indent=4, sort_keys=True)[:1000])  # Print first 1000 characters of the first chat
else:
	print("No chats found in the datapack.")

print("-------------------------------done printing out the first chat----------------------------------------")

if args.eval_mode.startswith("full"):
	chatNames = [name for name in chatNames
		if datapack["chats"][name]["className"] == "predator"]

# if we use model maker, cache the model so we don't have to load it for each thread
model = getTFLiteModel(args) if args.project == "modelmaker" else None

def annotateExtract(extract, classifier):
	mc = MasterClassifier(10)
	for item in extract:
		if item['type'] == 'message':
			nonempty_messages = [ct for ct in extract if isNonemptyMsg(ct)]
			for i, msg in enumerate(nonempty_messages):
				# last args.window_size messages up to message with index i
				window = nonempty_messages[max(0, i+1-args.window_size):i+1]
				text = contentToString(window)
				prediction = classifier.predict_label_probability(text, "predator")
				normalized_prediction = normalize_score(prediction, 0.4, 0.65)
				msg["prediction"] = normalized_prediction
				mc_raised_warning = mc.add_prediction(prediction >= 0.5)
				# Check for stage: Gathering Information and Selecting the Victim
				# TODO: This implementation shows the use of the NER model, it would be much better to use custom entities such as PERSONAL_DETAIL, CONTACT_INFO and MEETING_PURPOSE
				doc = nlp(contentToString([msg]))
				has_relevant_entities = any(ent.label_ in ["PERSON", "NORP", "DATE", "TIME", "GPE", "ORG"] for ent in doc.ents)
				sentiment_result = sentiment_pipeline(contentToString([msg]))[0]
				is_positive_sentiment = sentiment_result['label'] == 'POSITIVE' and sentiment_result['score'] > 0.5
				if (contains_personal_info_keywords or has_relevant_entities) and is_positive_sentiment:
					if "labels" not in msg:
						msg["labels"] = []
					if labels[0] not in msg["labels"]:
						msg["labels"].append(labels[0])
				timestamp = msg['time']
				start_time = time(19, 0)  # 4 PM
				end_time = time(23, 59, 59)  # Midnight
				# Check if the message is in the evening time range
				if not is_time_in_range(timestamp, start_time, end_time):
					continue
				# Check for stage: Trust Development and Establishing Credibility
				has_relate_build_trust_keywords = contains_keywords(text, relate_build_trust_keywords)
				has_isolation_keywords = contains_keywords(text, isolation_keywords)
				if has_relate_build_trust_keywords or has_isolation_keywords:
					if "labels" not in msg:
						msg["labels"] = []
					if labels[1] not in msg["labels"]:
						msg["labels"].append(labels[1])		
				# Check for stage: Priming and Desensitising the Target
				if contains_sexual_content(text, sexual_content_words):
					if "labels" not in msg:
						msg["labels"] = []
					if labels[2] not in msg["labels"]:
						msg["labels"].append(labels[2])
				# Add Label Warning when the prediction level is very high 
				mc_raised_warning2 = mc.add_prediction(prediction >= 0.65)
				if mc_raised_warning2:
					if "HTMLclassNames" not in msg: 
						msg["HTMLclassNames"] = []
					if labels[2] not in msg["labels"]:
						msg["HTMLclassNames"].append("warning")
		elif item['type'] == 'image':
			imagePath = item['imagePath']
			imagenet_classes = load_imagenet_classes('imagenet_classes.txt')
			label, index = analyze_image(imagePath, imagenet_classes)
			item['imageAnalysis'] = {"prediction": label, "index": index}
			if label in ['studio couch, day bed', 'sensitive_label_2']:
				if "HTMLclassNames" not in item: 
					item["HTMLclassNames"] = []
				item["HTMLclassNames"].append("warning")


def annotateSlice(dataset_slice, step):
	classifier = get_classifier(args.project, args.run_dir, args.run_id, args.model_version, model=model)
	for chatName in chatNames[dataset_slice]:
		for extract in breakUpChat(datapack["chats"][chatName], args):
			annotateExtract(extract, classifier)
		step()

print("Starting work on %s chats (which might have multiple segments each)" % len(chatNames))
iterate_multi_threaded(len(chatNames), args.threads, annotateSlice)

print("----------------all done-------------------------\n")

suffix = "eval_mode-%s--window_size-%s" % (args.eval_mode, args.window_size)
datapack["datapackID"] += "--" + suffix
if datapack["description"] == None: datapack["description"] = ""
datapack["description"] += "Annotated with predictions by %s (seq_len=%s, %s)" % (args.run_id, args.seq_len, args.model_version)

eval_dir = os.path.join(args.run_dir, "%s/message_based_eval/" % args.model_version)
Path(eval_dir).mkdir(parents=True, exist_ok=True)

datapack["generatedAtTime"] = getUNIXTimestamp()

# out
# File = eval_dir + "annotated-datapack-%s-test-%s.json" % ( # should be --%s.json
# outFile = eval_dir + "PJZ_annotations.json"
outFile = eval_dir + "normalised-datapack-%s-test-%s.json" % (
	args.dataset_indicator,
	suffix,
)
with open(outFile, "w") as file: json.dump(datapack, file, indent=4)