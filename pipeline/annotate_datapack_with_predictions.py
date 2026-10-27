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
from datetime import datetime, time
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
	"i know where", "remember", "white slut", "slut", "tell u", "you better", 'looking for a younger']

personal_info_keywords = [
    'where do you live', 'how old are you', 'old are you', 'old are u','what school', 'what grade','to old for u', 'too old for u', ' the oldest age u would date'
    'do u go to school', 'where is ur mom', 'u got a pic', ' where u frum', 'how old', 'do you like older guys?', 'the oldest u would date', 'can i ask a personal question first'
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
File = eval_dir + "annotated-datapack-%s-test-%s.json" % ( # should be --%s.json
# outFile = eval_dir + "PJZ_annotations.json"
outFile = eval_dir + "normalised-datapack-%s-test-%s.json" % (
	args.dataset_indicator,
	suffix,
)
with open(outFile, "w") as file: json.dump(datapack, file, indent=4)