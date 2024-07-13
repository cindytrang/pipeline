# 

from textblob import TextBlob
from transformers import pipeline
import pandas as pd

sentiment_pipeline = pipeline("sentiment-analysis")

    # extract sentiment scores
def add_textblob_sentiment_scores(df):
    df['textblob_sentiment'] = df['segment'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['textblob_label'] = df['textblob_sentiment'].apply(lambda x: 'POSITIVE' if x > 0 else ('NEGATIVE' if x < 0 else 'NEUTRAL'))
    return df

    # transformers sentiment score and label
def add_transformers_sentiment_scores(df):
    df['transformers_sentiment_score'] = df['segment'].apply(lambda x: sentiment_pipeline(x)[0]['score'])
    df['transformers_label'] = df['segment'].apply(lambda x: sentiment_pipeline(x)[0]['label'].upper())
    return df

data = {
    'segment': [
        "I'm so happy you're my friend!",
        "I'm worried about the test tomorrow.",
        "It's just a bad day, not a bad life.",
        "I hate waiting in lines."
    ]
}
df = pd.DataFrame(data)
df = add_textblob_sentiment_scores(df)
df = add_transformers_sentiment_scores(df)

print(df[['segment', 'textblob_label', 'textblob_sentiment', 'transformers_label', 'transformers_sentiment_score']])


import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

nlp = spacy.load('en_core_web_sm')

conversation ="""

darklil1nal (10:26:29 PM): hi
rod_blankenship2003 (10:26:44 PM): what you doin
darklil1nal (10:26:49 PM): nutin
darklil1nal (10:26:51 PM): asl?
darklil1nal (10:26:57 PM): 13 f bham
rod_blankenship2003 (10:27:10 PM): omg a baby(Yes. Now, go away.)
darklil1nal (10:27:21 PM): (silly smiley face here)
rod_blankenship2003 (10:27:39 PM): you sure are cute(Grooming begins here.)
darklil1nal (10:27:44 PM): ty
rod_blankenship2003 (10:28:06 PM): you know im alot older dont you
darklil1nal (10:28:11 PM): ah no
darklil1nal (10:28:15 PM): u didnt tel me
darklil1nal (10:28:16 PM): lol
rod_blankenship2003 (10:28:29 PM): well i just did ok
rod_blankenship2003 (10:28:31 PM): lol
darklil1nal (10:28:41 PM): lol
darklil1nal (10:28:53 PM): what lik 35?
rod_blankenship2003 (10:29:01 PM): yep is that ok
darklil1nal (10:29:06 PM): sure
rod_blankenship2003 (10:29:20 PM): are you the police (No, however I did tell them all about you.)
darklil1nal (10:29:30 PM): lol no
darklil1nal (10:29:50 PM): r u
rod_blankenship2003 (10:29:58 PM): well theyd love to see this conversation wouldnt they (They did.)
rod_blankenship2003 (10:30:34 PM): think i better go or do ya wanna talk (Thinking right there Rod, why did you stop thinking?)
darklil1nal (10:30:44 PM): ill chat sure
rod_blankenship2003 (10:30:56 PM): ok what ya want to talk about
darklil1nal (10:31:03 PM): u pick
rod_blankenship2003 (10:31:18 PM): nope you start
rod_blankenship2003 (10:31:27 PM): your gorgeous(More grooming.)
darklil1nal (10:31:50 PM): ty
darklil1nal (10:31:55 PM): u got pic?
rod_blankenship2003 (10:32:24 PM): no / 5''9 178 lt brwn hair blue eyes in good shape and a man not a boy
darklil1nal (10:32:31 PM): kewl
darklil1nal (10:32:39 PM): waht u do?
rod_blankenship2003 (10:32:59 PM): im kindoff a banker what do you do / lol
darklil1nal (10:33:08 PM): skewl silly
rod_blankenship2003 (10:33:26 PM): yeah i know / what grade / damn / lol
darklil1nal (10:33:39 PM): 8th
rod_blankenship2003 (10:33:50 PM): are you a virgin (Seven minutes.)
darklil1nal (10:34:06 PM): yeah mostly
rod_blankenship2003 (10:34:18 PM): do you want to change that
darklil1nal (10:34:38 PM): 1 day yeah
rod_blankenship2003 (10:34:49 PM): how soon / lol
darklil1nal (10:35:13 PM): lol
darklil1nal (10:35:20 PM): when i find some1 i like
rod_blankenship2003 (10:35:27 PM): hmmm how am i doin (Eight minutes.)
darklil1nal (10:35:46 PM): k so far
darklil1nal (10:35:46 PM): lol
rod_blankenship2003 (10:36:11 PM): really / where do you live babe(Assessement.)
darklil1nal (10:36:43 PM): bham
darklil1nal (10:36:44 PM): u
rod_blankenship2003 (10:37:03 PM): same / what part of bham you in
darklil1nal (10:37:25 PM): bessemer
darklil1nal (10:37:26 PM): u
rod_blankenship2003 (10:37:38 PM): leeds
darklil1nal (10:37:41 PM): kewl
rod_blankenship2003 (10:37:57 PM): your really not the police are you (So cautious, what are you afraid of Rod?)
darklil1nal (10:38:04 PM): ah no
darklil1nal (10:38:05 PM): r u
darklil1nal (10:38:07 PM): lol
rod_blankenship2003 (10:38:45 PM): no / would you call me for a minute
darklil1nal (10:38:54 PM): sure
darklil1nal (10:39:00 PM): i have 2 get off line
darklil1nal (10:39:07 PM): whats ur name
rod_blankenship2003 (10:39:16 PM): rod whats yours
darklil1nal (10:39:19 PM): liz
rod_blankenship2003 (10:39:27 PM): ya wanna call me( No, not really. However my verifier sure will call.)
darklil1nal (10:39:32 PM): sure
darklil1nal (10:39:35 PM): whats ur #
rod_blankenship2003 (10:39:48 PM): 790 1780 in about 5 minutes
darklil1nal (10:40:15 PM): ok then ill get back onlein
rod_blankenship2003 (10:40:23 PM): im kinda excited about it are you(Excited that a 13 year old is calling you? Why?)
darklil1nal (10:40:56 PM): yeah
rod_blankenship2003 (10:41:15 PM): good call me
darklil1nal (10:41:26 PM): ok let me get offline
darklil1nal (10:41:31 PM): and find the phone
rod_blankenship2003 (10:41:36 PM): let me add you first ok
darklil1nal (10:41:40 PM): k
darklil1nal (10:42:15 PM): ok let me get offline
rod_blankenship2003 (10:42:25 PM): good talk to ya in a minute
darklil1nal (10:55:45 PM): (more silly smilies)
rod_blankenship2003 (10:55:45 PM): hey
rod_blankenship2003 (10:55:57 PM): loved your voice
darklil1nal (10:56:01 PM): urs too
rod_blankenship2003 (10:56:19 PM): what should we do
darklil1nal (10:56:27 PM): do?
darklil1nal (10:56:40 PM): u tell me(I hate it when they want the kid to ask for it. Loathe it.)
rod_blankenship2003 (10:56:42 PM): well i like you
darklil1nal (10:57:07 PM): i like u 2
rod_blankenship2003 (10:57:32 PM): what are you thinkin (You, in an orange jumpsuit.)
darklil1nal (10:57:49 PM): well i dunt got no car
rod_blankenship2003 (10:58:08 PM): how is it we could get together
darklil1nal (10:58:27 PM): i gues i could sneak out
rod_blankenship2003 (10:58:53 PM): you live with your mom and dad right(More risk assessment.)
darklil1nal (10:58:58 PM): mom yeah
rod_blankenship2003 (10:59:08 PM): wheres dad
darklil1nal (10:59:22 PM): he left
rod_blankenship2003 (10:59:45 PM): omg i loved your little voice (She sounds 12 Sicko)
darklil1nal (11:00:21 PM):(silly smiley)
rod_blankenship2003 (11:01:20 PM): tell me what your thinkin now baby(You in an orange jumpsuit with lots of others in orange jumpsuits.)
darklil1nal (11:01:33 PM): nervous
darklil1nal (11:01:40 PM): he he
darklil1nal (11:01:46 PM): u wont hurt me wil u
darklil1nal (11:03:50 PM): u there
rod_blankenship2003 (11:04:28 PM): did i offend you
darklil1nal (11:04:32 PM): no
darklil1nal (11:04:35 PM): sorry i got booted
rod_blankenship2003 (11:05:02 PM): its ok
rod_blankenship2003 (11:05:33 PM): you ok
darklil1nal (11:05:46 PM): yeah
darklil1nal (11:05:47 PM): u
rod_blankenship2003 (11:05:58 PM): yeah im kinda happy
rod_blankenship2003 (11:06:20 PM): you live in bessemer(Lots and lots of risk assessment.)
darklil1nal (11:06:24 PM): why
rod_blankenship2003 (11:06:41 PM): just curious
darklil1nal (11:06:50 PM): oh ok
rod_blankenship2003 (11:07:10 PM): im begining to feel im scaring you should i go (Yes, run for the hills Rod! Run.)
darklil1nal (11:07:19 PM): no
darklil1nal (11:07:25 PM): tel me about u
darklil1nal (11:07:29 PM): im just nervour
darklil1nal (11:07:31 PM): thats all
darklil1nal (11:07:36 PM): sowwy
rod_blankenship2003 (11:07:47 PM): would you call me back in a little while
rod_blankenship2003 (11:08:09 PM): im 5''9 178 lt brwn hair blue eyes and in good shape is that ok
darklil1nal (11:08:28 PM): well i need to go to bed
darklil1nal (11:08:30 PM): i got skewl
darklil1nal (11:08:32 PM): tomorrow
darklil1nal (11:08:39 PM): can i call tomorro mebee
rod_blankenship2003 (11:08:46 PM): ok and i have to go to work
darklil1nal (11:08:56 PM): k
rod_blankenship2003 (11:09:13 PM): sure ya can call tomorrow can ya call around 9 in the evening
darklil1nal (11:09:18 PM): kewl
rod_blankenship2003 (11:09:32 PM): will ya think about me tonight(I will be thinking about the detectives you will meet.)
darklil1nal (11:09:41 PM): u sound nice
darklil1nal (11:09:46 PM): wish i had a pic
rod_blankenship2003 (11:09:49 PM): so did you
rod_blankenship2003 (11:09:59 PM): i loved your voice
darklil1nal (11:10:08 PM): he he
rod_blankenship2003 (11:10:27 PM): why do you want to loose your virginity to an older man(Puke.)
darklil1nal (11:10:51 PM): boys my age ar bum
darklil1nal (11:10:53 PM): dum
rod_blankenship2003 (11:11:04 PM): i know / can i ask some sexual questions (No.)
darklil1nal (11:11:21 PM): tomorrow
darklil1nal (11:11:23 PM): its late
rod_blankenship2003 (11:11:36 PM): ok baby / ya goin to bed now
darklil1nal (11:12:02 PM): nite nite
rod_blankenship2003 (11:12:05 PM): would you like for me to be with you
darklil1nal (11:12:07 PM): yes
darklil1nal (9:35:56 PM): hey
rod_blankenship2003 (9:36:16 PM): hey girl
darklil1nal (9:36:25 PM): sowwy i havent ben on
darklil1nal (9:36:28 PM): i got grounded(Could not stand the thought of talking to you again.)
rod_blankenship2003 (9:36:33 PM): where ya been
darklil1nal (9:37:09 PM): i got gfounded
rod_blankenship2003 (9:37:14 PM): uh oh what did ya do
rod_blankenship2003 (9:37:54 PM): ya there
darklil1nal (9:38:05 PM): made a d on a test
rod_blankenship2003 (9:38:20 PM): what subject
darklil1nal (9:38:26 PM): mth
rod_blankenship2003 (9:38:48 PM): you need a boyfriend dont ya / lol (That will help my grades how?)
darklil1nal (9:38:54 PM): lol
darklil1nal (9:38:55 PM): yeah
rod_blankenship2003 (9:39:06 PM): i know this older guy ya might like / lol
darklil1nal (9:39:10 PM): u?
darklil1nal (9:39:13 PM): lol
rod_blankenship2003 (9:39:21 PM): yep
rod_blankenship2003 (9:39:33 PM): i missed not talkin to you
darklil1nal (9:39:43 PM): sowwy
rod_blankenship2003 (9:39:54 PM): its silly but i really like you(It is not silly, it is sick and perverted.)
darklil1nal (9:40:05 PM): i like u 2
rod_blankenship2003 (9:40:11 PM): thanks baby
rod_blankenship2003 (9:40:25 PM): we were gonna talk about sex remember(Good thing I did not have dinner tonight.)
rod_blankenship2003 (9:40:26 PM): lol
darklil1nal (9:40:30 PM): lol
rod_blankenship2003 (9:40:36 PM): want too
darklil1nal (9:40:41 PM): i gues sur
rod_blankenship2003 (9:40:48 PM): if ya dont i wont ok
darklil1nal (9:41:01 PM): its ok
rod_blankenship2003 (9:41:08 PM): have you kissed a boy yet
darklil1nal (9:41:12 PM): yes
rod_blankenship2003 (9:41:18 PM): did ya like it
darklil1nal (9:41:21 PM): yes
rod_blankenship2003 (9:41:44 PM): did ya kinda tingle in your panties when ya did it
darklil1nal (9:41:53 PM): lol yeah
rod_blankenship2003 (9:42:01 PM): lol i like you baby
rod_blankenship2003 (9:42:06 PM): your fun
rod_blankenship2003 (9:42:24 PM): have you fantasized about having sex
darklil1nal (9:42:37 PM): not realy
rod_blankenship2003 (9:42:40 PM): no
rod_blankenship2003 (9:42:45 PM): hmmmm
rod_blankenship2003 (9:42:52 PM): question
rod_blankenship2003 (9:43:10 PM): if you and i kissed could i feel of ya
darklil1nal (9:43:22 PM): sure
rod_blankenship2003 (9:43:34 PM): i mean in your panties
darklil1nal (9:44:06 PM): i gues
rod_blankenship2003 (9:44:09 PM): do you play with it sometime
rod_blankenship2003 (9:44:22 PM): tell the truth / lol
darklil1nal (9:44:32 PM): not realy
rod_blankenship2003 (9:44:42 PM): so you dont think of sex much
rod_blankenship2003 (9:45:06 PM): am i makin you mad or uncomfortable baby(Yes, very mad.)
darklil1nal (9:45:21 PM): i havent done much
darklil1nal (9:45:30 PM): so i dnot now wat to think about
darklil1nal (9:45:31 PM): lol
rod_blankenship2003 (9:46:11 PM): whats the most youve done
darklil1nal (9:46:23 PM): kissin
rod_blankenship2003 (9:46:30 PM): and ya liked it
darklil1nal (9:46:34 PM): rubin
rod_blankenship2003 (9:46:41 PM): rubin where
darklil1nal (9:46:51 PM): u now
darklil1nal (9:46:52 PM): lol
rod_blankenship2003 (9:47:00 PM): his dick
darklil1nal (9:47:05 PM): yeah
rod_blankenship2003 (9:47:11 PM): how old was he
darklil1nal (9:47:17 PM): 25
rod_blankenship2003 (9:47:21 PM): no way
rod_blankenship2003 (9:47:29 PM): where did you meet him
darklil1nal (9:47:36 PM): lol
darklil1nal (9:47:41 PM): he went out with mom
rod_blankenship2003 (9:47:58 PM): and he played with you when she wasnt there
darklil1nal (9:48:07 PM): yes
rod_blankenship2003 (9:48:19 PM): did you like it
darklil1nal (9:48:29 PM): i gues yeah
rod_blankenship2003 (9:48:40 PM): did you get it out and rub it
darklil1nal (9:48:49 PM): no
rod_blankenship2003 (9:48:55 PM): just under his pants
darklil1nal (9:48:57 PM): yes
rod_blankenship2003 (9:49:03 PM): did he rub you in your panties
darklil1nal (9:49:08 PM): yes
rod_blankenship2003 (9:49:14 PM): did he finger you
darklil1nal (9:49:17 PM): no
rod_blankenship2003 (9:49:28 PM): just played with it
rod_blankenship2003 (9:49:57 PM): did he want to do more
darklil1nal (9:50:05 PM): yeah
darklil1nal (9:50:08 PM): mom woke u
darklil1nal (9:50:14 PM): thouhg
rod_blankenship2003 (9:50:17 PM): was he in bed with you
darklil1nal (9:50:27 PM): on teh couch
rod_blankenship2003 (9:50:37 PM): did you kiss him
darklil1nal (9:51:09 PM): is tht what u wanta do to me
rod_blankenship2003 (9:51:14 PM): did you get up after your mom went to sleep
rod_blankenship2003 (9:51:41 PM): and get on the couch with him(No, this is not one of those sick fantasies where the child seduces you.)
darklil1nal (9:51:46 PM): no
rod_blankenship2003 (9:52:21 PM): tell me how you got with him then ok(Oh you sick, sick old man.)
darklil1nal (9:52:39 PM): lol
darklil1nal (9:52:43 PM): y
darklil1nal (9:52:50 PM): tell me waht u wanta do
rod_blankenship2003 (9:53:15 PM): what would you like me to do honey
darklil1nal (9:53:21 PM): i duno
darklil1nal (9:53:27 PM): u the grown up
darklil1nal (9:53:27 PM): lol
rod_blankenship2003 (9:53:39 PM): did you like playin with him on the couch
darklil1nal (9:53:46 PM): sure
rod_blankenship2003 (9:53:57 PM): did you want him to fuck you
darklil1nal (9:54:01 PM): no
rod_blankenship2003 (9:54:05 PM): why not(Ah, because I was less than 13 years old at the time.)
darklil1nal (9:54:09 PM): is that waht u wanta do
darklil1nal (9:54:15 PM): i was scard
darklil1nal (9:54:19 PM): mom wuld catch us
rod_blankenship2003 (9:54:21 PM): how old is your mom
darklil1nal (9:54:32 PM): 34
rod_blankenship2003 (9:54:38 PM): do you want me to date her so we can see each other( Just UGH! Who would go this far? This is just UGH!)
darklil1nal (9:54:49 PM): no
rod_blankenship2003 (9:54:59 PM): what do you want us to do
darklil1nal (9:55:07 PM): i duno hang out
darklil1nal (9:55:08 PM): and stuf
rod_blankenship2003 (9:55:15 PM): how can we do that baby
darklil1nal (9:55:29 PM): i duno
rod_blankenship2003 (9:55:41 PM): would your mom let you see a man my age
darklil1nal (9:55:49 PM): lol no
rod_blankenship2003 (9:55:56 PM): didnt think so
rod_blankenship2003 (9:56:12 PM): does she still see the guy you were on the couch with
darklil1nal (9:56:16 PM): no
rod_blankenship2003 (9:56:23 PM): did he tell ya not to tell anyone
darklil1nal (9:56:27 PM): yes
rod_blankenship2003 (9:56:36 PM): lmao ill bet he did
rod_blankenship2003 (9:56:51 PM): do you know he could go to prison for that(And you for this Rod.)
darklil1nal (9:56:59 PM): realy?
darklil1nal (9:57:01 PM): no
rod_blankenship2003 (9:57:24 PM): hell yes / your underaged dear thats called a sexual preditor and child abuse(Just what you are planning there Rod. Look in the mirror to see a predator.)
darklil1nal (9:57:29 PM): wow
darklil1nal (9:57:41 PM): r u a predator
rod_blankenship2003 (9:57:45 PM): nope (That should be YES. Take a closer look in the mirror.)
rod_blankenship2003 (9:57:56 PM): are you the police
rod_blankenship2003 (9:57:58 PM): lol
darklil1nal (9:57:59 PM): lol no
rod_blankenship2003 (9:58:19 PM): well if your not tell me what youd like me to do to you
darklil1nal (9:58:32 PM): i duno teach me i gues
darklil1nal (9:58:38 PM): i duno much
rod_blankenship2003 (9:58:47 PM): well what did he teach you
darklil1nal (9:58:53 PM): not much
rod_blankenship2003 (9:59:07 PM): how long did you stay on the couch with him

"""

conversation1 ="""
hi
how u 2 nite?
great!!!
good
whats up?
 i dont no yet
 i jus got here
u liv in chicago area?
 ya
where?
 at my house! ( Hey, I'm 11 what do I know? )
lol
im in western subburbs
 k
when school start 4 u ?
 tues
wow
 do u go to school?
summer over already
 ya
what grade or yr u in?
 i will be in 6th
ok

 do u like horses?
yes
u ride?
 i want to
 my mom says no
u hav a horse?
 no
any pets?
 ya
dog?
 kitty

u like kitty?
 ya
 she is a very good kitty
 her name is baxter
does she listen?
 ya she listesns
how old is she?
 she is 2 i think
does ur sister like the cat? ( This is classic pervert strategy - he is digging for information on how many family members are around to assess his risk )
 i dont have a sister
brother?
 no just me
ur parents divod?
 u mean divorsed?
divorced
 ya
u liv w/mom
 ya how did u no it?
just guess
u miss having a sis or brother? ( And this is classic wannabe pedophile information digging - he is looking for a child who is isolated and likely to be in need of attention. That need is what he wants to exploit )
 i dont no i never had one silly
lol

wheres mom now? ( Again, how much can he get away with? )
 waching a movie
u like talking on computor?
 ya
 do u?
yes
 i like to see what the chat sez
 sometimes
u hav cam?
 no
 i wish!!
 is that u??? ( He sent me the link to watch him on camera. It's a live shot of him fondling himself through a pair of grey sweat shorts )
 wow
u like 2 watch?
 ya
 u went away ( He got booted offline here but came back )
u always watch?
 i never did see that
not even in person
 u wont tell?
no
 i seen one b4
when u baby sit?
 no
 my mom had a bf
 he showed it
 and how to make it go
did u touch it?
 ya
 he let me
 r u mad??
 r u mad atme? ( This guy kept popping in and out of attention, which I later realized was simply a matter of him having some techinical difficulties. Everytime he comes back the webcam showing him fondling himself is offered to me again )

 u r not doing it
hello
 hi
i got disconnected
 o
 k
 wb

where is ur moms b/f now?
 her new one??
the 1 that showed u?
 oh they broke up
she has a new 1 now?
 ya
does he show u?
 no
 he is new
u ever see ur mom ..havin sex? ( This is this particular pervert's obsession )
 no
 but if it is noizy i no
is it noisy?
 now?
no no
when mom has sex?
 sometimes
u see ur mom naked?
 sometimes
how old is she?
 i dont no
 35 maybe
u ever see her pussy hair?
 ya
what color is it?
 yellow
blonde
 ya
u wanna touch my cock? ( And here comes the agenda )
 ya
 u let me??
yes

u ever see it cum? ( Had I known at this point where he would go with that idea I would have killed that camera. BLECH! )
 ya
u make it cum?
 ya
how?
 i put my hands there
thats all?
 no
what else?
 u wont tell
no
cause i want u 2 do it 2 me
ok

what else?
 i kissed it
any thing else?
 i have to go pp k?
pp?
u there
 ya
 had to pp
ok
u wanna meet me tomorrow? ( Yeah. In a dark alley with a group of goombahs and a baseball bat. Hey, I can dream! )
 k

where u liv?
what town?
 chicago
 or by what main streets in chicago? ( At first he was very thrilled to have a child alone at the computer for his show, but now that he has gotten the child into dirty talk so agreeably, he wants to make it real )
 is there a busy street u know of? ( At this point I am dead on confident this wannabe pedophile believes without any doubt he is dealing with a real eleven year old girl )
 we live by the school
 what school?
 the univerty
 can u call me on the phone now? ( Ahhhhh here we go )
 ya
 what university?
 university of chicago???
 on south side?
 we r in hide park
 yes i know where that is

 can u meet me by the museum?
 which one?
 science & industry

 thats near hyde park?
 ya
 i can ride my bike there
 if i dont tell my mom i am goin
 ????
 some other guy is sayin his thing is hard
 to me
 tell him 2 get lost ( Awww, isn't that sweet? He's afraid some other guy might get a chance to rape the kids of his dreams first )
 ur not interested
 k
 708 227 9428 my phone # call me ( Yes, call him. Call him now. Call him later. Call him all day long )
 rite now?
 yes
 i have to sneak the phone
 k?
 k
 will u let me touch ur thing?
 yes

 u will show me how
 will u kiss me?
 ya
 we get naked 2 gether
 k
 i will show u my thing
 2

 ok

 i am goin to get the phone
 ok
 i dont no ur name to call
 jim
 jim
 my name is brittany
 ok
 k
 i am goin to go call
 ok
 a lady anserd!!!!!!!!!
 u dialed wrong #
 what is it agin?
 u r not a lday rite?
 1 708 2279428
 k
 1 708 227 9428
 no
 im male
 k
 u r grown up and drive ya?
 yes
 Folks, let me just tell you here, I realized I don't have his picture. The profile has an obscene cartoon, I see him on camera, but no pic for the file. The following gets really SICK. I put up with this trying to get a picture out of this miserable piece of wannabe pedophile shit. Fair warning, the whole time he is jacking off on his camera, which I have covered by other stuff. So when he completes it and shows off what he "made" I realized I wasn't ever going to get a head shot out of him. )
 i no becuz u have a big thing!!!
 can i see ur thing agin when I talk to u???
yes

 do u have apic?
 so i will no u?
u gonna call?
 ya
u there?
hello
u didnt call me
 ino
 my mom is doin stuff in the kichen
 so i have to b quiet
 sory
ok
 im not suposed to b up
past ur bed time
 ya
listen listen
make sure u delete this conversation
 k
 my mom doesnt no id o yahoo
what is ur phone #
u hav bathin suit?
 ya
 i got a bikinni
wear it 2 morrow under ur cloths

ok
 k
 will we go swimmin?
yes
 wow!!!!!!
 i luv 2 swim
can i see u naked?
 k
i kiss ur pussy
 k

stick my dick in it 2?
 will it hurt???
no
 k
u hav any hair on ur pussy?
 no
 is that ok?
yes
u hav any tits yet? ( Just a reminder of exactly how young this girl is supposed to be. )
 not realy
 they r small
ok
 is that ok?
 u will still like me?
yes

u a police officer? ( He's been so caught up in believing he has a shot at nabbing a real child he almost forgot about the law )
 no
 silly
 my mom is a actres
u hav anything 2 do with cops or police or law enforcemaent?
 no y?
 do u want 2 be a police?
i dont want 2 show up
then go 2 jail
 ?
 r u a bad guy?
im no cop
no
nice guy ( Sure, just a very nice wannabe pedophile looking to abduct and have sex with an eleven year old girl. Real nice, buddy )
 good
 k

 i dont like robbers
im not
 k
 one time
 a robber took my moms vcr
 so i dont like them
how did he get in the house 2 take it?
 i dont no
where is ur dad at?
 texas
ok
 will u make that stuff in your thing?
u wanna watch other guys jerk off? ( This offer makes me believe this guy belongs to a newsgroup or online group of pedophiles. He certainly sounds like he knows how to arrange a show for his buddies, no? )
 what guys?
on line?
 idont no
 ?
u there?

call me
 i cant yet
 i think they made popcorn
1 708 227 9428 ( Okay, that is the FOURTH time, so I'm calling this a solid number. )
 k
 can i see u so i cna see if u r cute? ( I'm still hoping for a photo. He has been jacking off on camera the whole time. I have that image buried under other pms and stuff, but I have to check it now and then to see if I can see his face. No dice. )
who made pop corn?
 my mom and her bf
 downstaris
u gonna watch them hav sex? ( fetish anyone? )
 i dont no if they r
 u want to?
do u ?
 k
 if i listen i can tell if htey do
i'll teach u how 2 make it cum ( What a generous offer, however will I resist?? )

u like that?
 ya
 i got another picture to show u ( I'm still trying to warm him up into trading pics )
 u want to see it?
ok send it
 u will see my cowboy hat
yes very pretty

 u like my hat?
when was that taken? ( the photo is obviously from around a year prior to the one in my profile, which would make the child in the photo 9 or 10 years old at the time )
 last year
yes

wear ur hat tommorrow
ok
 i cant it is at my dads
ok
 i got another one
 want to see it?
yes
 k
u hav 1 in a swim suit
u hav 1 in a swim suit ( His obsession with little kids in swimsuits makes me think this pervert probably hangs out at local pools. Look for the single white guy in the corner staring at the littlest girls. Ick )
 i dont no
 i dont think so
 on my computre
u like lookin at naked boys ur age? ( here comes a lot of stall chatter from me, keeping him on the line )
 they have small things
yes
but do they show u?
 one time
 but it wasno big one
just u saw
 ya
or ur g/friends 2 saw it
 how did u no?
 at camp
how many girls saw it?
 a lot like 5?
 im thinkin whowas there
all at 1 time?
 me an ashley kelly emily
 ya
did any 1 touch it?
 no
just 1 boy showed it?
 no
 3 boys
were u naked 2?
 no
did u girls laugh?
 no
 kenny peed his
 so we could see
u saw a boy go pee?
 ya
did u laugh?
 no
 its cool how boys do it
thats all u did was look at them?
 ya
 then i ask john to show me when he pees
 so he did
how old were the boys?
 i dont no 10?
how old were u ?
 10
that was 2 yr ago?
 1
 last year
 in the summer
not this yr?
 no i didnt go this summer
u like lookin at girls naked?
 k
what did u do this summer?
 went to californa with my mom
thats all?
 oh no
 i did lots of stuff
 swimmin and gymnastics
 i got a baton
did u see boys naked at swimming? ( Oh yeah, the kids at the local pools are all butt naked. Sure. )
 no
 th eonly boy i saw naked here is john
alone in ur house?
 ya
 my mom was at work
did u get naked 2?
 ya
what did u do?
 u wont tell???
no
 i asked him to show me his thing
 when he pees
how old is john?
 i dont no
 he was my moms bf
ok
did he show u?
 ya
 he let me hold it
did u kiss & touch it?
 ya ( during this chatter Mr. Sickwit here is wanking himself but good. He is also rocking a bit in his chair when he does it. I put up with this crap hoping he would lean back far enough for me to see his face. If I weren't so distracted by a sense of a mission I probably would have hurled during this. Just reading it now for editing is making me queasy. )
 when i hold it it gets bigger
u like that?
 ya
did it cum?
 ya
u touch the cum?
 ya
u taste the cum?
 he put it on my mouth with his hand
u like the taste?
 its ok
u got naked 4 him?
 ya
 when it got hard he todl me to
what did he do 2 u ?
 he showd me if you put ur hand on it
 and go fast it gets big
 and then if u kiss it u no?
 u can make it cum
did u play with his balls?
 not realy
 very much
did he kiss ur pussy?
 no
 he rubbed it
 hello??
hello
he didnt kiss ur pussy
 no
u watchin another guy jerk off ( Ironically, this bastard is NOT the only one trying to show me his jack off skills on camera. I was turning away other wannabe pedophiles left and right during this chat )
 no
u want me?
 ya
 will u smile at me on ur camera??
r u gettin excited watchin me?
 ya
ur mom works during the day? ( More attempts to establish a timeframe for opportunities to molest this kid )
 sometimes
u home alone?
 sometimes
we can meet during the day ?
 ya
know one will no
 no
 u want to come to my house?
when u know me
 k
 u r touching it
we can play in ur bed ( Okay, let's play "Strap the pervert to it and set the whole thing on fire" )
 ya

 it is pink

u c my cum
 wow!!!!!!!! ( This is where I just about puked. This guy jacked off into his hand and put the handful up into the camera shot. For an eleven year old. )
u like 2 touch it
 ya
u gonna call me 2 morrow? ( No, but I hope every one of our readers does )
 ya!
u gonna go some where with ur mom ?
"""

conversation2 = """
hi, 51/guy/fayetteville, wanna talk?
sadlilgrrl (4:41:20 PM): sure--r u okay with me bein almost 14? ( Blam. There's the age, RIGHT THERE. )
arthinice (4:41:36 PM): sure ( Not even a pause. )
arthinice (4:41:39 PM): how are you?
sadlilgrrl (4:41:55 PM): im pretty good
arthinice (4:42:07 PM): pretty good too
arthinice (4:42:11 PM): what are you up to today?
sadlilgrrl (4:42:25 PM): bored
arthinice (4:42:38 PM): was your spring break this past week?
sadlilgrrl (4:42:53 PM): i dunno. im homeschooled til august
arthinice (4:43:04 PM): what happens in august?
sadlilgrrl (4:43:11 PM): i can start the new year
arthinice (4:43:35 PM): homeschooling? or public school?
sadlilgrrl (4:43:41 PM): public
sadlilgrrl (4:43:51 PM): i moved in february
sadlilgrrl (4:43:53 PM): n it was too late
arthinice (4:43:59 PM): where did you move from?
sadlilgrrl (4:44:19 PM): nc
arthinice (4:44:29 PM): where did you move to?
sadlilgrrl (4:44:33 PM): chicago
arthinice (4:44:43 PM): is that where you are now?
sadlilgrrl (4:44:50 PM): yah
sadlilgrrl (4:44:58 PM): during the summer
sadlilgrrl (4:45:02 PM): but 4 now im in lr
sadlilgrrl (4:45:08 PM): stayin with my gramma
arthinice (4:45:11 PM): i see
arthinice (4:45:19 PM): i bet you are a pretty girl (Dan here is a groomer. You'll note this.)
arthinice (4:45:25 PM): do you have a pic?
sadlilgrrl (4:45:41 PM): yah
arthinice (4:45:45 PM): (emoti)
arthinice (4:45:51 PM): i'd love to see - if you don't mind
arthinice (4:46:13 PM): mine is on my profile if you wanna see it
sadlilgrrl (4:46:18 PM): ok ill look
arthinice (4:46:43 PM): i see 2 very pretty girls
sadlilgrrl (4:46:45 PM): you look really nice
sadlilgrrl (4:46:47 PM): im onthe right
arthinice (4:46:57 PM): who is the other girl?
sadlilgrrl (4:47:10 PM): my friend from nc
arthinice (4:47:28 PM): is she 14 too?
sadlilgrrl (4:47:33 PM): yah
arthinice (4:47:43 PM): wow - both of you are really pretty!!
sadlilgrrl (4:47:52 PM): tyvm!
arthinice (4:47:54 PM): you seem to be a very sweet girl ( No idea how he knows that. )
sadlilgrrl (4:47:59 PM): i like to think i am
sadlilgrrl (4:48:00 PM): lol
arthinice (4:48:09 PM): do you have a bf?
sadlilgrrl (4:48:46 PM): no
sadlilgrrl (4:48:56 PM): i havent been in lr too long
arthinice (4:49:15 PM): you are too pretty for that to last too long
sadlilgrrl (4:49:39 PM): i dunno
sadlilgrrl (4:49:45 PM): the guys i know are pretty dumb
arthinice (4:50:14 PM): yeah - guys your age are kinda dumb
arthinice (4:50:18 PM): have you ever had a bf?
sadlilgrrl (4:50:22 PM): yeah
sadlilgrrl (4:50:25 PM): back in nc
arthinice (4:50:30 PM): how old was he?
sadlilgrrl (4:50:35 PM): 19
arthinice (4:50:41 PM): wow - older guy
arthinice (4:50:55 PM): do you like older guys?
sadlilgrrl (4:50:58 PM): yeah
sadlilgrrl (4:51:04 PM): they arent obsessed w/video games
sadlilgrrl (4:51:04 PM): lol
arthinice (4:51:09 PM): (emoti)
arthinice (4:51:19 PM): no - they are obsessed with pretty girls
sadlilgrrl (4:51:24 PM): which is better
arthinice (4:51:35 PM): pretty girls - of course!!
arthinice (4:51:57 PM): how long were you together?
sadlilgrrl (4:52:07 PM): 4 months
arthinice (4:52:13 PM): then you had to move
arthinice (4:52:14 PM): ?
sadlilgrrl (4:52:18 PM): yeah
arthinice (4:52:26 PM): do you still keep in touch with him?
sadlilgrrl (4:52:35 PM): no, not really.
arthinice (4:52:45 PM): you look sooo sweet ( Steady stream of compliments. )
sadlilgrrl (4:52:50 PM): i am.
sadlilgrrl (4:52:51 PM): (emoti)
arthinice (4:52:52 PM): i love your smile
arthinice (4:53:09 PM): the one in the pic is prettier than the smiley on here!
arthinice (4:53:36 PM): do you have any more pix?
sadlilgrrl (4:53:39 PM): yeah
sadlilgrrl (4:53:40 PM): a couple
arthinice (4:53:55 PM): the only one i have is on my profile
arthinice (4:54:04 PM): but i'd love to see yours - if you don't mine
arthinice (4:54:55 PM): is your name kristen?
sadlilgrrl (4:54:58 PM): yeah
arthinice (4:55:42 PM): very pretty name
arthinice (4:55:49 PM): can you hold for a min - brb
sadlilgrrl (4:55:52 PM): k
arthinice (4:56:54 PM): sorry about that - potty break ( He delibrately uses language that a young kid would here. )
sadlilgrrl (4:56:59 PM): that's ok.
sadlilgrrl (4:57:01 PM): (emoti)
arthinice (4:57:31 PM): wow - you look sooo pretty in this pic!!
sadlilgrrl (4:57:38 PM): tyvm!
arthinice (4:58:02 PM): who is the guy that looks like he is being tortured??
sadlilgrrl (4:58:31 PM): lol
sadlilgrrl (4:58:35 PM): hes hillarys lil bro
arthinice (4:58:48 PM): he looks like he is in pain - poor guy
sadlilgrrl (4:59:12 PM): lol like i said guys my age are dumb
arthinice (4:59:23 PM): little does he know how much pleasure he could get from a pretty girl sitting on his lap
sadlilgrrl (4:59:45 PM): lol
arthinice (4:59:46 PM): you didn't kiss him - did you? ( What a sleazy little question. )
sadlilgrrl (4:59:53 PM): no
sadlilgrrl (4:59:54 PM): lol
sadlilgrrl (4:59:57 PM): just my ex br
sadlilgrrl (4:59:59 PM): bf*
arthinice (5:00:05 PM): he would have died!!!
sadlilgrrl (5:00:11 PM): probably.
sadlilgrrl (5:00:13 PM): (emoti)
arthinice (5:00:21 PM): you kissed your ex bf?
sadlilgrrl (5:00:26 PM): yeah
sadlilgrrl (5:00:29 PM): when he was my bf
arthinice (5:00:37 PM): i bet you have really sweet kisses ( *mutters* )
sadlilgrrl (5:00:51 PM): he didnt complain
sadlilgrrl (5:00:53 PM): (emoti)
arthinice (5:01:27 PM): i'm sure i wouldn't either ( I'm sure you wouldn't. )
sadlilgrrl (5:01:35 PM): as is good and proper! ( AIGAP indeed. )
sadlilgrrl (5:01:36 PM): lol
arthinice (5:02:00 PM): do you have another pic?
sadlilgrrl (5:02:06 PM): two
arthinice (5:02:10 PM): cool
arthinice (5:02:19 PM): i hope you don't mind me seeing them
sadlilgrrl (5:02:25 PM): no your really nice
sadlilgrrl (5:02:28 PM): n your kinda cute
sadlilgrrl (5:02:29 PM): (emoti)
arthinice (5:02:40 PM): you are pumping my ego now
sadlilgrrl (5:02:46 PM): no way
arthinice (5:02:52 PM): but don't stop!!
sadlilgrrl (5:03:04 PM): ok
sadlilgrrl (5:03:04 PM): lol
sadlilgrrl (5:03:05 PM): (emoti)
arthinice (5:03:27 PM): its not often a pretty young girl pumps my ego - it feels pretty good ( And as you'll see, that's not all he wants pumped. )
sadlilgrrl (5:03:37 PM): lol well ill be happy to.
sadlilgrrl (5:03:51 PM): did u get it?
arthinice (5:03:54 PM): yes
arthinice (5:03:59 PM): you really do look tired
sadlilgrrl (5:04:02 PM): lol i was!
arthinice (5:04:06 PM): looks like you stayed up all night
sadlilgrrl (5:04:11 PM): wed been up all night at the sleepover
sadlilgrrl (5:04:11 PM): lol
sadlilgrrl (5:04:12 PM): yah
arthinice (5:04:20 PM): buncha girls there?
sadlilgrrl (5:04:28 PM): yeah
sadlilgrrl (5:04:31 PM): it was at hillarys house
sadlilgrrl (5:04:35 PM): n then we went to the countryclub
sadlilgrrl (5:04:41 PM): n jennys my other best friend
arthinice (5:04:49 PM): cool
arthinice (5:05:22 PM): i hope you are not bored now
sadlilgrrl (5:05:29 PM): not at all
arthinice (5:05:32 PM): wow - 3 very very pretty girls
sadlilgrrl (5:05:35 PM): lol
arthinice (5:06:10 PM): so - what else are you gonna say to pump my ego?
sadlilgrrl (5:06:30 PM): lol well your funny
sadlilgrrl (5:06:31 PM): and nice
sadlilgrrl (5:06:32 PM): and cute
sadlilgrrl (5:06:35 PM): your a winner! ( You've won a chat with a PJ contributor! )
arthinice (5:06:42 PM): wow - a winner?
sadlilgrrl (5:06:54 PM): yeah!
arthinice (5:06:57 PM): i've not been called a winner in a very very long time
arthinice (5:06:59 PM): thank you!!
arthinice (5:07:18 PM): you are sooo sweet
arthinice (5:07:23 PM): i bet your bf misses you
sadlilgrrl (5:07:27 PM): eh.
sadlilgrrl (5:07:32 PM): hell find someone else
arthinice (5:07:44 PM): but she won't be quite the same
sadlilgrrl (5:07:50 PM): he didnt seem to mind too much
arthinice (5:08:10 PM): is he the only guy you've kissed so far?
sadlilgrrl (5:08:15 PM): yeah
arthinice (5:08:46 PM): i'm probably out of line to say so - but i sure wish i could taste some of your sweet kisses too ( Yes. )
sadlilgrrl (5:08:53 PM): y would you be out of line?
sadlilgrrl (5:08:57 PM): hold on brb
arthinice (5:09:10 PM): potty break?
sadlilgrrl (5:10:03 PM): yah
sadlilgrrl (5:10:04 PM): lol
arthinice (5:10:10 PM): (emoti)
arthinice (5:10:48 PM): do you think you'd like for a guy my age to taste some of your kisses sometime? ( And here it begins. )
sadlilgrrl (5:10:56 PM): i bet your a good kiser
sadlilgrrl (5:10:57 PM): kisser
arthinice (5:11:04 PM): i like to kiss
arthinice (5:11:13 PM): and i like cuddles and hugs too
sadlilgrrl (5:11:20 PM): me too
arthinice (5:11:41 PM): i bet you would be fun to cuddle with and hug ( Indeed. )
sadlilgrrl (5:11:47 PM): i like to think so
arthinice (5:12:05 PM): can i ask a really personal question?
sadlilgrrl (5:12:17 PM): sure
arthinice (5:12:27 PM): are you a virgin?
sadlilgrrl (5:12:33 PM): yah ( This is not a sexually aware child at all. He's going to work on that. )
sadlilgrrl (5:12:42 PM): i was ready but then i had to move
arthinice (5:13:06 PM): how far did you get to go with your bf?
arthinice (5:13:10 PM): any touching?
sadlilgrrl (5:13:34 PM): yeah
sadlilgrrl (5:13:36 PM): lol
sadlilgrrl (5:13:37 PM): (emoti)
arthinice (5:13:42 PM): did you like it?
sadlilgrrl (5:13:56 PM): yeah
sadlilgrrl (5:14:01 PM): he wasnt very gentle all the time tho
arthinice (5:14:09 PM): (emoti)
arthinice (5:14:15 PM): i like to be very gentle
arthinice (5:14:22 PM): girls are too fragile
sadlilgrrl (5:14:26 PM): (emoti)
arthinice (5:14:29 PM): soft and tender
sadlilgrrl (5:14:43 PM): yes
arthinice (5:15:02 PM): you look sooo soft and tender
sadlilgrrl (5:15:10 PM): i use lotion everyday! ( Hahaha. I made myself laugh here. )
arthinice (5:15:13 PM): (emoti)
arthinice (5:15:26 PM): did he touch your nipples? ( Because asking that's appropriate. )
sadlilgrrl (5:15:35 PM): yeah once
arthinice (5:15:43 PM): did he kiss them? ( See above comment. )
sadlilgrrl (5:15:57 PM): no
arthinice (5:16:05 PM): i love to kiss nipples
sadlilgrrl (5:16:12 PM): does it feel good?
arthinice (5:16:12 PM): i bet yours are wonderful to kiss
arthinice (5:16:29 PM): it feels good to me - i'd hope it would feel too to you too
sadlilgrrl (5:16:39 PM): probably
arthinice (5:16:52 PM): of course the feeling i get will be different from the feeling you get
sadlilgrrl (5:17:04 PM): why?
arthinice (5:17:32 PM): if the guys does it right - you should get at tiingly feeling "down there"
sadlilgrrl (5:17:51 PM): i got that when he touched me
arthinice (5:17:57 PM): for me - it just tastes good and feels good in my mouth
sadlilgrrl (5:17:58 PM): on my breasts
arthinice (5:18:17 PM): it was a good tingly feeling - right?
sadlilgrrl (5:18:30 PM): yeah
arthinice (5:18:47 PM): did he touch you "down there" too? ( Don't worry, I get vocab lessons later.)
sadlilgrrl (5:18:55 PM): yeah
sadlilgrrl (5:18:59 PM): just that 1 time tho
arthinice (5:19:08 PM): did you like how that felt?
sadlilgrrl (5:19:24 PM): yeah
sadlilgrrl (5:19:26 PM): (emoti)
arthinice (5:19:28 PM): (emoti)
arthinice (5:19:41 PM): that's where a guy really has to be soft and gentle
sadlilgrrl (5:19:49 PM): yeah
sadlilgrrl (5:19:54 PM): he wasnt too terribly
arthinice (5:20:10 PM): i bet he didn't touch it for very long either - right?
sadlilgrrl (5:20:16 PM): yeah
arthinice (5:20:35 PM): did he insert his finger into you? ( Well, from here on out, it gets graphic. Just read it. If I bolded everything, it'd look a mess. I'll bold highlights, so to speak. )
sadlilgrrl (5:20:44 PM): no
arthinice (5:21:03 PM): did you get to touch him too?
sadlilgrrl (5:21:05 PM): yeah
sadlilgrrl (5:21:08 PM): it was pretty cool
arthinice (5:21:24 PM): did you just touch it? or did you do anything else?
sadlilgrrl (5:21:29 PM): i rubbed it upa nd down
arthinice (5:21:34 PM): (emoti)
arthinice (5:21:41 PM): did you make it really hard?
sadlilgrrl (5:21:44 PM): yeah
sadlilgrrl (5:21:45 PM): lol
sadlilgrrl (5:21:48 PM): that wa s the fun part
arthinice (5:22:17 PM): did you kiss it?
sadlilgrrl (5:22:20 PM): no
arthinice (5:22:47 PM): did you want to?
sadlilgrrl (5:22:54 PM): sort of
sadlilgrrl (5:23:02 PM): hillary told me guys like that
arthinice (5:23:20 PM): has she kissed some?
sadlilgrrl (5:23:27 PM): i dunno
sadlilgrrl (5:23:30 PM): i think 1
arthinice (5:23:42 PM): is this making you uncomfortable? ( He knows this is wrong.)
sadlilgrrl (5:23:47 PM): nah its ok
sadlilgrrl (5:23:49 PM): your really nice
arthinice (5:23:56 PM): am i still a winner??
sadlilgrrl (5:23:59 PM): you are!!!
arthinice (5:24:02 PM): cool
arthinice (5:24:06 PM): you are just too sweet
arthinice (5:24:15 PM): so - you rubbed it up and down
sadlilgrrl (5:24:25 PM): yeah
arthinice (5:24:28 PM): did anything happen?
sadlilgrrl (5:24:35 PM): it got hard
arthinice (5:24:43 PM): anything else?
sadlilgrrl (5:24:50 PM): it got wet kind of
arthinice (5:25:05 PM): a little bit of clear stuff came out?
sadlilgrrl (5:25:08 PM): yeah
arthinice (5:25:39 PM): anything else?
sadlilgrrl (5:25:57 PM): not until it kind of shook n stuff came otu
sadlilgrrl (5:25:58 PM): out
arthinice (5:26:10 PM): it kind of shook?
sadlilgrrl (5:26:15 PM): well he did
arthinice (5:26:34 PM): do you think he liked it?
sadlilgrrl (5:26:45 PM): i think so
sadlilgrrl (5:26:46 PM): lol
sadlilgrrl (5:26:49 PM): he said it was good
arthinice (5:26:51 PM): i think he did too
sadlilgrrl (5:26:55 PM): n i coul practice on him more
arthinice (5:27:09 PM): did you get to practice on him more?
sadlilgrrl (5:27:12 PM): no
sadlilgrrl (5:27:14 PM): i had to move
sadlilgrrl (5:27:15 PM): (emoti)
arthinice (5:27:43 PM): when he shook - what happened?
sadlilgrrl (5:27:49 PM): he kind of fell onto the sofa
sadlilgrrl (5:27:52 PM): n said i did a good job
sadlilgrrl (5:27:55 PM): n stuff came out
arthinice (5:28:03 PM): white stuff?
sadlilgrrl (5:28:06 PM): yeah
sadlilgrrl (5:28:12 PM): come
sadlilgrrl (5:28:19 PM): he said
arthinice (5:28:30 PM): yes - some spell it cum
sadlilgrrl (5:28:39 PM): yeah ive seen it spelled that way too
sadlilgrrl (5:28:46 PM): i hafta go pick up my room n start dinner
sadlilgrrl (5:28:49 PM): will u be here in a bit?
arthinice (5:28:56 PM): yes - how long will you be gone?
sadlilgrrl (5:29:05 PM): probably like 30 minutes or so
arthinice (5:29:09 PM): ok
arthinice (5:29:14 PM): i sure like talking with you
sadlilgrrl (5:29:16 PM): ok ill talk to you then
sadlilgrrl (5:29:18 PM): i like talkin to you too
sadlilgrrl (5:29:20 PM): ur nice to me
arthinice (5:29:24 PM): (emoti)
arthinice (5:29:28 PM): please hurry - ok?
sadlilgrrl (5:29:31 PM): k

I wandered off. Cleaned the house. Came back.

sadlilgrrl (6:10:03 PM): back
arthinice (6:10:10 PM): (emoti)
arthinice (6:10:13 PM): i missed you
sadlilgrrl (6:10:15 PM): you're still here!
sadlilgrrl (6:10:16 PM): yay!
arthinice (6:10:27 PM): am i still a winner??
sadlilgrrl (6:10:31 PM): you ARE!
arthinice (6:10:48 PM): cool
arthinice (6:10:56 PM): how much longer can you chat?
sadlilgrrl (6:11:01 PM): a while
sadlilgrrl (6:11:04 PM): mom's at work
sadlilgrrl (6:11:14 PM): she called n said she wouldnt be home for a while still
sadlilgrrl (6:11:19 PM): after i started dinner
arthinice (6:11:19 PM): ok
sadlilgrrl (6:11:20 PM): (emoti)
arthinice (6:11:33 PM): what were we talking about?
sadlilgrrl (6:11:48 PM): you said you liked takling t me
sadlilgrrl (6:11:52 PM): and i like dtalking to you
arthinice (6:11:58 PM): yes
arthinice (6:12:11 PM): i keep looking at your pix
arthinice (6:12:20 PM): golly - you are soooo pretty!!!
sadlilgrrl (6:12:40 PM): lol
sadlilgrrl (6:12:43 PM): your making me blush
arthinice (6:12:50 PM): good
sadlilgrrl (6:13:00 PM): (emoti)
arthinice (6:13:18 PM): i think we were talking about you rubbing your bf's cock - right? ( That's what he wants to talk about. )
sadlilgrrl (6:13:26 PM): oh yeah
arthinice (6:13:50 PM): and all that white stuff coming out
sadlilgrrl (6:13:52 PM): yeah
sadlilgrrl (6:13:58 PM): there was kind of a lot
arthinice (6:14:09 PM): did it just shoot in the air?
sadlilgrrl (6:14:26 PM): yeah n on my han
sadlilgrrl (6:14:28 PM): hand
arthinice (6:14:45 PM): was it sticky?
sadlilgrrl (6:14:49 PM): yeah it was
arthinice (6:15:02 PM): did you like seeing it come out?
sadlilgrrl (6:15:04 PM): it tasted kind of salty too
sadlilgrrl (6:15:06 PM): yah it was cool
arthinice (6:15:23 PM): you tasted it?
sadlilgrrl (6:15:31 PM): yeah he said hed tasted it and it tasted good
arthinice (6:15:47 PM): can i tell you what some girls like to do?
sadlilgrrl (6:15:50 PM): what?
arthinice (6:15:59 PM): sure you want me to tell you?
sadlilgrrl (6:16:02 PM): yeah
arthinice (6:16:12 PM): its not gross - at least i don't think it is
sadlilgrrl (6:16:15 PM): okay
arthinice (6:16:25 PM): some girls like to "kiss" it
sadlilgrrl (6:16:34 PM): yeah it seems like thatd be kind of fu
sadlilgrrl (6:16:35 PM): fun
arthinice (6:16:45 PM): have you seen it - or heard about it?
sadlilgrrl (6:17:20 PM): yeah hillary told me about it a little bit
arthinice (6:17:44 PM): you know that purple part at the tip of the cock?
sadlilgrrl (6:17:48 PM): yeah
arthinice (6:17:59 PM): girls will put that part in their mouth
sadlilgrrl (6:18:08 PM): do the guys like it?
arthinice (6:18:12 PM): uh huh
sadlilgrrl (6:18:24 PM): cool
arthinice (6:18:29 PM): it feels really good
sadlilgrrl (6:18:30 PM): do you like it?
arthinice (6:18:38 PM): uh huh
sadlilgrrl (6:18:41 PM): thats cool
arthinice (6:18:59 PM): some girls can get the whole shaft in their mouth
sadlilgrrl (6:19:07 PM): wow
arthinice (6:19:09 PM): not just the purple tip
sadlilgrrl (6:19:11 PM): that must tkae practice
arthinice (6:19:16 PM): yes
arthinice (6:19:22 PM): wanna practice?? ( Just, eww. )
sadlilgrrl (6:19:31 PM): i wanna be good at it
arthinice (6:19:42 PM): i bet you will be very good at it
arthinice (6:19:58 PM): when you are good at it...
arthinice (6:20:13 PM): you will know how to make it go in and out of your mouth
sadlilgrrl (6:20:27 PM): thats cool
arthinice (6:20:29 PM): kinda like your hand rubbing it up and down
arthinice (6:20:34 PM): know what i mean?
sadlilgrrl (6:20:46 PM): i think so
arthinice (6:21:14 PM): and then - what do you think you should do when it shakes and all the cum comes out?
sadlilgrrl (6:21:20 PM): i dunno
sadlilgrrl (6:21:58 PM): what am i supposed to do?
arthinice (6:22:02 PM): well - some girls take the cock out of their mouth and let it squirt on their face
arthinice (6:22:19 PM): or just in the air
sadlilgrrl (6:22:27 PM): oh
arthinice (6:22:33 PM): how does that sound?
sadlilgrrl (6:22:37 PM): messy
sadlilgrrl (6:22:38 PM): lol
arthinice (6:22:41 PM): uh huh
arthinice (6:22:51 PM): some other girls do something different
sadlilgrrl (6:22:55 PM): what?
arthinice (6:23:00 PM): can you guess?
sadlilgrrl (6:23:14 PM): eat it?
arthinice (6:23:34 PM): well - kinda - swallow it
sadlilgrrl (6:23:42 PM): thats probably less messy
arthinice (6:23:51 PM): yes
arthinice (6:23:55 PM): what do you think about that?
sadlilgrrl (6:24:05 PM): its smarter
arthinice (6:24:07 PM): you've tasted it
arthinice (6:24:13 PM): think you could swallow it?
sadlilgrrl (6:24:21 PM): probably
arthinice (6:24:36 PM): (emoti)
arthinice (6:24:48 PM): can i tell you about something else?
sadlilgrrl (6:24:52 PM): okay
arthinice (6:25:13 PM): you said younger guys are dumb
arthinice (6:25:30 PM): and when it comes to sex - they are REALLY dumb
arthinice (6:25:57 PM): most young guys only want to be satisfied
sadlilgrrl (6:26:01 PM): okay
arthinice (6:26:09 PM): they have no clue how to satisfy a girl
arthinice (6:26:41 PM): they think that if they just touch a girl a little that somehow she is satisfied
arthinice (6:26:47 PM): but that is not true
sadlilgrrl (6:26:51 PM): oh
arthinice (6:26:58 PM): can i tell you more?
sadlilgrrl (6:27:07 PM): sure
arthinice (6:27:17 PM): sure you wanna hear this stuff??
sadlilgrrl (6:27:20 PM): yeah!
sadlilgrrl (6:27:23 PM): im not a little kid
arthinice (6:27:27 PM): i know
arthinice (6:27:45 PM): your bf is a classic example of what i'm talking about
sadlilgrrl (6:27:58 PM): yeah
arthinice (6:27:59 PM): it kinda felt good for him to touch you but...
arthinice (6:28:17 PM): you didn't "shake" like he did - right?
sadlilgrrl (6:28:46 PM): no
arthinice (6:28:50 PM): did you know that you can "shake" too?
sadlilgrrl (6:29:19 PM): not really
arthinice (6:29:27 PM): do you know what it is called?
sadlilgrrl (6:29:34 PM): coming?
arthinice (6:29:50 PM): yes - but there is another name for it too
arthinice (6:30:26 PM): it is usually talked about happening to girls - but the same name applies to guys too
sadlilgrrl (6:30:32 PM): oh
sadlilgrrl (6:30:33 PM): what is it?
arthinice (6:30:35 PM): it is called orgasm ( See? Vocab lessons. )
sadlilgrrl (6:30:40 PM): oh okay
arthinice (6:30:44 PM): have you heard of that?
sadlilgrrl (6:31:33 PM): i think so.
arthinice (6:31:42 PM): do you remember how he shook?
sadlilgrrl (6:31:55 PM): yeah
arthinice (6:31:57 PM): ok
arthinice (6:32:11 PM): and remember how you felt tingly when he touched your nipples
sadlilgrrl (6:32:13 PM): yeah
arthinice (6:32:14 PM): and your pussy? ( He introduces that word. )
sadlilgrrl (6:32:19 PM): uhhuh
arthinice (6:32:26 PM): felt good - huh
arthinice (6:32:38 PM): well - if he knew what he was doing...
sadlilgrrl (6:32:42 PM): yeah kind of
arthinice (6:33:01 PM): and he did it long enough - that tingly feeling would turn into a "shake"
arthinice (6:33:06 PM): or orgasm
sadlilgrrl (6:33:20 PM): oh
arthinice (6:33:29 PM): can i make you promise me something?
sadlilgrrl (6:34:03 PM): okay
arthinice (6:34:36 PM): are you there?
sadlilgrrl (6:34:43 PM): yeah i said okay
arthinice (6:34:51 PM): next time you are with a guy...
arthinice (6:35:09 PM): and he wants you to make him cum...
arthinice (6:35:21 PM): you tell him you will be glad to make him cum...
arthinice (6:35:33 PM): if he will make you cum first
sadlilgrrl (6:35:36 PM): lol
sadlilgrrl (6:35:36 PM): okay
arthinice (6:35:37 PM): ok?
sadlilgrrl (6:35:51 PM): i bet your good about that, huh?
arthinice (6:35:54 PM): do you know why i want you to promise that?
sadlilgrrl (6:36:13 PM): cuz it will feel good?
arthinice (6:36:24 PM): yes - for sure - but there is something else too
arthinice (6:36:32 PM): remember - guys are dumb
sadlilgrrl (6:36:47 PM): ok
arthinice (6:36:57 PM): if you don't say that to a guy - he will never satisfy you
sadlilgrrl (6:37:09 PM): oh
arthinice (6:37:10 PM): he will only be interested in being satisfied
arthinice (6:37:15 PM): understand?
sadlilgrrl (6:37:18 PM): yeah
arthinice (6:37:30 PM): now - please understand...
arthinice (6:37:40 PM): i'm talking about most guys - not all guys
arthinice (6:37:54 PM): if you are very very lucky...
arthinice (6:38:17 PM): you will find a guy that is more concerned with your satisfaction than his
arthinice (6:38:31 PM): but don't hold your breath
sadlilgrrl (6:38:34 PM): lol
sadlilgrrl (6:38:35 PM): ok
arthinice (6:38:45 PM): does all that make sense?
sadlilgrrl (6:38:56 PM): yeah
arthinice (6:39:10 PM): now - let me ask another question...
arthinice (6:39:24 PM): are you being selfish to ask him to satisfy you first?
sadlilgrrl (6:39:30 PM): not if i do it back
arthinice (6:39:35 PM): right!!
arthinice (6:39:39 PM): good girl!!!
arthinice (6:39:54 PM): but what if you satisfy him first? what will happen then?
sadlilgrrl (6:40:06 PM): he wont do anything to me?
arthinice (6:40:12 PM): you are not only sweet
arthinice (6:40:14 PM): and pretty
arthinice (6:40:17 PM): and sexy
arthinice (6:40:25 PM): you are one very smart girl too!!! ( Groomer. )
sadlilgrrl (6:40:34 PM): (emoti)
sadlilgrrl (6:40:38 PM): (emoti)
arthinice (6:40:49 PM): now - can i tell you the best senario?
sadlilgrrl (6:40:53 PM): ok
arthinice (6:41:19 PM): what would it be like if both of you satisfied each other at the same time?
sadlilgrrl (6:41:44 PM): i dunno if i could think very well
arthinice (6:41:51 PM): (emoti)
arthinice (6:42:09 PM): there are different ways to do it - but that is the best way
arthinice (6:42:19 PM): both kissing each other
arthinice (6:42:27 PM): touching each other
arthinice (6:42:37 PM): feeling each other
arthinice (6:42:46 PM): god - i'm making myself horny!!!
sadlilgrrl (6:43:12 PM): like wanting it?
arthinice (6:43:16 PM): uh huh
sadlilgrrl (6:43:21 PM): yeah
sadlilgrrl (6:43:23 PM): me to kidna
sadlilgrrl (6:43:31 PM): kinda
arthinice (6:43:37 PM): remember how his cock got hard?
sadlilgrrl (6:43:43 PM): yeah
arthinice (6:43:47 PM): mine is too ( Good to know. )
sadlilgrrl (6:43:57 PM): thats awesome!!!1
arthinice (6:44:11 PM): do you ever get a little wet "down there"?
sadlilgrrl (6:44:22 PM): a little bit
arthinice (6:44:46 PM): i bet if we were together - kissing and cuddling and touching and feeling...
arthinice (6:44:56 PM): i bet it would be more than a little bit ( Not that he wants to do that. )
sadlilgrrl (6:45:00 PM): probably
sadlilgrrl (6:45:06 PM): you sound like your really good and nice
arthinice (6:45:23 PM): can i tell you something i like to do?
sadlilgrrl (6:45:32 PM): okay
arthinice (6:45:41 PM): your legs look sooo sexy
arthinice (6:46:02 PM): i like to kiss the soft inner part of sexy legs - the part just above your knees
arthinice (6:46:12 PM): know where i'm talking about?
sadlilgrrl (6:46:32 PM): the thigh?
arthinice (6:46:35 PM): uh huh
arthinice (6:46:41 PM): do you have shorts on?
sadlilgrrl (6:47:03 PM): yeah
sadlilgrrl (6:47:14 PM): mesh onse
sadlilgrrl (6:47:16 PM): ones
arthinice (6:47:30 PM): if you take your finger and gently run it up your thigh from your knee up...
arthinice (6:47:44 PM): you will feel a little bit of how it would feel to be kissed there ( He's educational! )
sadlilgrrl (6:47:53 PM): it tickles kind of
arthinice (6:47:58 PM): uh huh
arthinice (6:48:00 PM): do you like it?
sadlilgrrl (6:48:13 PM): yeah
arthinice (6:48:16 PM): (emoti)
arthinice (6:48:33 PM): and then i like to kiss all the way to the top of the thigh
sadlilgrrl (6:48:53 PM): oh that sounds nice
sadlilgrrl (6:48:56 PM): (emoti)
arthinice (6:49:02 PM): all the way up to where your panties are now
arthinice (6:49:30 PM): but if i kissed your thighs like that...
arthinice (6:49:45 PM): i'd carefully pull your panties and shorts off
arthinice (6:49:53 PM): would you let me?
sadlilgrrl (6:49:55 PM): well you couldnt reach my thighs with them on
arthinice (6:50:18 PM): good point
arthinice (6:50:32 PM): so - would you let me take them off?
sadlilgrrl (6:50:43 PM): yeah
sadlilgrrl (6:50:45 PM): i guess so
arthinice (6:50:46 PM): (emoti)
sadlilgrrl (6:50:50 PM): im kinda curious
sadlilgrrl (6:50:52 PM): you know?
sadlilgrrl (6:50:54 PM): is that bad?
arthinice (6:50:55 PM): uh huh
sadlilgrrl (6:50:56 PM): im sorry
arthinice (6:51:00 PM): not not bad at all
arthinice (6:51:16 PM): but only if you want to
sadlilgrrl (6:51:25 PM): i kinda do
arthinice (6:51:29 PM): if you said no - or stop - i would not do it
arthinice (6:51:38 PM): i would honor your request
arthinice (6:52:05 PM): have you felt your pussy before?
sadlilgrrl (6:52:29 PM): yeah
arthinice (6:52:47 PM): know how it has that silky slit in the middle?
arthinice (6:53:31 PM): the part between the pussy lips? ( A description. Great. )
sadlilgrrl (6:53:31 PM): yeah
arthinice (6:53:48 PM): i'd want to stroke my tongue up and down that part ( I bet you would. )
sadlilgrrl (6:54:00 PM): oh
sadlilgrrl (6:54:03 PM): does that feel good?
arthinice (6:54:05 PM): especially if it is really wet there
arthinice (6:54:16 PM): feel good to me? or to you?
sadlilgrrl (6:54:29 PM): either?
arthinice (6:54:37 PM): i like the feel and the taste
arthinice (6:54:47 PM): but it is you that would be feeling good
arthinice (6:54:54 PM): that tingly feeling
sadlilgrrl (6:55:04 PM): cool
arthinice (6:55:04 PM): but a lot lot lot stronger
arthinice (6:55:47 PM): think you'd like that?
sadlilgrrl (6:56:28 PM): i think so
arthinice (6:56:38 PM): i bet you'd really like it
arthinice (6:57:06 PM): and if i did it long enough and was careful not to hurt you
arthinice (6:57:23 PM): in a while your body would shake just like your bf's did
sadlilgrrl (6:57:52 PM): that sounds really nice
arthinice (6:58:03 PM): what else do you think you'd like?
sadlilgrrl (6:58:09 PM): i dont know.
sadlilgrrl (6:58:15 PM): i want to try like all the way
sadlilgrrl (6:58:19 PM): but i want it to feel good
arthinice (6:58:28 PM): think it will?
sadlilgrrl (6:58:36 PM): it probaby depends on the guy
arthinice (6:58:42 PM): exaclty
arthinice (6:58:52 PM): too the words right out of my mouth!!
sadlilgrrl (6:59:03 PM): (emoti)
arthinice (6:59:20 PM): kristen - can i tell you a little secret?
sadlilgrrl (6:59:28 PM): okay
arthinice (6:59:36 PM): its not a dark deep secret or anything scary
sadlilgrrl (6:59:46 PM): i trust you
arthinice (6:59:59 PM): i wish i could be the one to show you how good all this feels ( He gets more explicit. )
sadlilgrrl (7:00:13 PM): you really do sound like you know what youre doing
arthinice (7:00:41 PM): well - we'd get to see if i am still a winner or not - huh ( He really fixated on that. Hahahaha. )
sadlilgrrl (7:00:45 PM): lol yah
arthinice (7:01:02 PM): let me ask you another question
sadlilgrrl (7:01:06 PM): okay
sadlilgrrl (7:01:08 PM): (emoti)
arthinice (7:01:22 PM): when you go "all the way"...
arthinice (7:01:36 PM): the guy puts his cock into your pussy - right?
sadlilgrrl (7:02:05 PM): yeah
arthinice (7:02:40 PM): and he pushes it in and pulls it out - kinda like you rubbing your hand up and down it - right?
arthinice (7:03:20 PM): after a while he is gonna "shake" again an cum is gonna come out
sadlilgrrl (7:03:30 PM): oh
arthinice (7:03:46 PM): what happens if he squirts his cum inside you? what could happen? ( Education, part 2. )
sadlilgrrl (7:03:56 PM): i could get preggers
sadlilgrrl (7:03:58 PM): right?
arthinice (7:04:01 PM): yep
arthinice (7:04:13 PM): so what do you do to keep that from happening?
sadlilgrrl (7:04:36 PM): use condoms
arthinice (7:04:44 PM): yes - that is one way
arthinice (7:04:52 PM): what else could you do?
sadlilgrrl (7:05:00 PM): be on bitrh control
arthinice (7:05:10 PM): yes - anything else?
sadlilgrrl (7:05:24 PM): pull out
sadlilgrrl (7:05:26 PM): right?
arthinice (7:05:35 PM): yes - how did you know that??
sadlilgrrl (7:05:51 PM): hillary
sadlilgrrl (7:05:51 PM): lol
arthinice (7:05:52 PM): you are one very smart girl!!
arthinice (7:06:02 PM): she is one very smart girl too!!
arthinice (7:06:15 PM): which of those 3 is the most risky?
sadlilgrrl (7:06:25 PM): i dont know
arthinice (7:06:37 PM): condom, pull out or birth control?
sadlilgrrl (7:06:52 PM): the condom?
arthinice (7:06:55 PM): nope
arthinice (7:06:58 PM): pull out
sadlilgrrl (7:07:01 PM): oh
arthinice (7:07:04 PM): know why?
sadlilgrrl (7:07:18 PM): no
arthinice (7:07:38 PM): cause there is no feeling in the world like shooting hot cum into pussy ( Wouldn't know. )
sadlilgrrl (7:07:46 PM): oh
arthinice (7:07:49 PM): when i am in there and it is time to cum
arthinice (7:08:00 PM): the last thing i wanna do is pull out
arthinice (7:08:14 PM): and even if i do...
arthinice (7:08:31 PM): i may not pull out soon enough - and may squirt a little inside
sadlilgrrl (7:08:46 PM): oh
arthinice (7:08:51 PM): make sense?
sadlilgrrl (7:08:56 PM): yeah
sadlilgrrl (7:09:00 PM): so its better to wear a condom?
arthinice (7:09:08 PM): or birth control
arthinice (7:09:31 PM): can i tell you another little secret?
sadlilgrrl (7:09:39 PM): okay
sadlilgrrl (7:09:42 PM): (emoti)
arthinice (7:09:48 PM): most guys - including me - HATE condoms
arthinice (7:09:53 PM): they don't feel right
sadlilgrrl (7:09:55 PM): so birth control then?
arthinice (7:09:58 PM): not natural
arthinice (7:10:22 PM): i wish i could give you an example
arthinice (7:10:38 PM): know how it feels when you touch your nipples?
sadlilgrrl (7:10:41 PM): yeah
arthinice (7:10:49 PM): your soft fingers on your nipple? ( As opposed to my hard fingers, I suppose. )
sadlilgrrl (7:10:55 PM): yeah
arthinice (7:11:14 PM): now imagine touching them with those rubber gloves doctors have to wear
sadlilgrrl (7:11:19 PM): taht woudl suck
sadlilgrrl (7:19:06 PM): im sorry i crashed
arthinice (7:19:14 PM): whew!!
arthinice (7:19:21 PM): thought it was something i said
sadlilgrrl (7:19:24 PM): no!!!
sadlilgrrl (7:19:39 PM): i like you
arthinice (7:19:46 PM): i like you too
sadlilgrrl (7:20:06 PM): (emoti)
arthinice (7:20:14 PM): i really wish we didn't live so far apart
sadlilgrrl (7:20:19 PM): how far are you?
arthinice (7:20:31 PM): about 3 hrs drive time
sadlilgrrl (7:20:51 PM): oh
sadlilgrrl (7:20:57 PM): thats too far, huh?
arthinice (7:21:17 PM): well - not really all that far - i just never get down that way
arthinice (7:21:29 PM): have you heard of fayetteville?
sadlilgrrl (7:21:57 PM): yea
sadlilgrrl (7:21:58 PM): yeah
arthinice (7:22:05 PM): that's where i am
arthinice (7:22:14 PM): opps - another potty break - brb
sadlilgrrl (7:22:27 PM): okay
arthinice (7:23:16 PM): back
sadlilgrrl (7:23:24 PM): yay!
sadlilgrrl (7:23:28 PM): i wish you werent so far away
arthinice (7:23:43 PM): and even if i did get to come down there - i don't know how we'd get to see each other
sadlilgrrl (7:23:54 PM): lol that wouldnt be hard
arthinice (7:24:05 PM): really? how?
sadlilgrrl (7:24:24 PM): my gramma's only home for the first part of the day
sadlilgrrl (7:24:26 PM): she works nights
sadlilgrrl (7:24:31 PM): thats why im online
arthinice (7:24:37 PM): ok
arthinice (7:24:49 PM): how long will you be in LR?
sadlilgrrl (7:25:02 PM): til end of may
arthinice (7:25:06 PM): dang
sadlilgrrl (7:25:16 PM): unless i tell my mom i like it here more
arthinice (7:25:18 PM): i'm gonna be pretty close to LR in june
arthinice (7:25:56 PM): june is a long way off but...
sadlilgrrl (7:26:08 PM): i dont wanna wait that long
sadlilgrrl (7:26:09 PM): lol
arthinice (7:26:21 PM): if we keep chatting and getting to know each other - maybe we can see each other
arthinice (7:26:33 PM): you want it now??
sadlilgrrl (7:26:42 PM): lol yeah, kind of
arthinice (7:26:43 PM): did i make you horny too??
sadlilgrrl (7:26:48 PM): yeah
arthinice (7:27:04 PM): sure wish i could be there with you right now
sadlilgrrl (7:27:20 PM): me too.
arthinice (7:27:27 PM): when is your b'day?
sadlilgrrl (7:28:08 PM): april 7
arthinice (7:28:19 PM): pretty close
arthinice (7:28:25 PM): you will be 14?
sadlilgrrl (7:28:37 PM): yeah
arthinice (7:29:00 PM): i wish i could give you a very special present - know what i mean?
sadlilgrrl (7:29:11 PM): letting me practice?
arthinice (7:29:30 PM): uh huh
sadlilgrrl (7:30:01 PM): i wish you could too
sadlilgrrl (7:30:03 PM): thatd be awesome
arthinice (7:30:05 PM): on your profile it says you are going to ny for your b'day
sadlilgrrl (7:30:12 PM): yeah but i dont know if i get to now or not
arthinice (7:30:28 PM): that sounds like fun too
sadlilgrrl (7:31:02 PM): yeah.
sadlilgrrl (7:31:06 PM): except my gramma might not be able to go
sadlilgrrl (7:31:08 PM): and shed have to work
sadlilgrrl (7:31:10 PM): n then i couldnt go
arthinice (7:31:14 PM): (emoti)
arthinice (7:31:32 PM): is it just you and gramma where you live?
sadlilgrrl (7:31:37 PM): yeah
arthinice (7:31:53 PM): where is mom and dad? can i ask that?
sadlilgrrl (7:32:08 PM): moms in chicago
sadlilgrrl (7:32:11 PM): i dont know where dad is
arthinice (7:32:37 PM): so - you're gonna go live with your mom this summer?
sadlilgrrl (7:32:51 PM): maybe
arthinice (7:33:13 PM): can i ask you a silly little question?
sadlilgrrl (7:33:23 PM): okay
arthinice (7:33:40 PM): your nick is sadlilgrrl - are you still a sad little girl?
sadlilgrrl (7:33:47 PM): not right now
arthinice (7:33:50 PM): (emoti)
sadlilgrrl (7:33:54 PM): your really nice
sadlilgrrl (7:33:59 PM): i wish you could come n see me
arthinice (7:34:05 PM): i wish i could too
arthinice (7:34:40 PM): but, kristen - you know that i could get in a lot lot lot of trouble doing what we talked about today - right? ( He knows it's wrong. )
sadlilgrrl (7:34:53 PM): why?
arthinice (7:35:01 PM): i mean - if we got caught
sadlilgrrl (7:35:09 PM): oh
sadlilgrrl (7:35:12 PM): i wouldnt tell
sadlilgrrl (7:35:15 PM): i promise
arthinice (7:35:47 PM): i know - i just want you to be aware
arthinice (7:36:09 PM): in fact - your ex bf could have gotten into lot of trouble too
sadlilgrrl (7:36:53 PM): oh
arthinice (7:37:09 PM): guys over 18 are not supposed to have sex with girls under 18
sadlilgrrl (7:37:25 PM): thats stupid
arthinice (7:37:42 PM): maybe - but it is the law
sadlilgrrl (7:37:51 PM): oh
sadlilgrrl (7:37:57 PM): so i guess you dont want to talk to me anymore
arthinice (7:38:13 PM): no, baby - i did not mean for you to think that
arthinice (7:38:19 PM): i love talking with you!!!
sadlilgrrl (7:38:27 PM): are you sure?
arthinice (7:38:37 PM): yes - of course
sadlilgrrl (7:38:39 PM): okay
arthinice (7:38:51 PM): i just want you to be aware of what could happen
arthinice (7:39:14 PM): just like all the things i explained about sex - i want you to be aware ( So educational. So good to me. )
sadlilgrrl (7:39:20 PM): okay
arthinice (7:39:44 PM): understand?
sadlilgrrl (7:39:47 PM): yeah
arthinice (7:39:58 PM): cool
arthinice (7:40:48 PM): so - do you think you can wait til june for m?
arthinice (7:40:49 PM): me?
sadlilgrrl (7:41:06 PM): my mom probably wouldnt care if i stayed down here
arthinice (7:41:16 PM): cool
sadlilgrrl (7:41:41 PM): would you really meet me?
arthinice (7:41:47 PM): yes
arthinice (7:41:51 PM): i would love to
sadlilgrrl (7:41:52 PM): thatd be so cool
sadlilgrrl (7:42:21 PM): you wont tell my gramma or anything will you?
arthinice (7:42:29 PM): god no!!!
arthinice (7:42:38 PM): she could have me put in jail!!
sadlilgrrl (7:42:49 PM): okay
sadlilgrrl (7:42:54 PM): i dont wanna get in trouble or get you in trouble
sadlilgrrl (7:42:56 PM): your too nice
sadlilgrrl (7:43:02 PM): and smart
arthinice (7:43:14 PM): and still a winner?
sadlilgrrl (7:43:19 PM): STILL a winner!!
sadlilgrrl (7:43:21 PM): (emoti)
arthinice (7:43:25 PM): no one has ever called me a winner ( Really fixated on that damn winner comment. )
arthinice (7:43:28 PM): i like it
sadlilgrrl (7:44:00 PM): i have kind of a weird ?
arthinice (7:44:06 PM): ok
sadlilgrrl (7:44:15 PM): whats your voice sound like?
arthinice (7:44:25 PM): an old man
sadlilgrrl (7:44:29 PM): lol
sadlilgrrl (7:44:30 PM): i duobt that
sadlilgrrl (7:44:32 PM): doubt
arthinice (7:44:43 PM): i bet your voice is so sweet
sadlilgrrl (7:44:52 PM): lol want me to call you?
arthinice (7:45:10 PM): i don't want you to get in trouble
sadlilgrrl (7:45:25 PM): my grammas at work n i have a phone card for calls n stuff
arthinice (7:45:33 PM): (emoti)
arthinice (7:45:51 PM): i won't be able to talk for very long - just a few minutes - ok?
sadlilgrrl (7:45:54 PM): okay
sadlilgrrl (7:45:59 PM): the phone might die neways
arthinice (7:46:09 PM): i know your first name
arthinice (7:46:14 PM): did i ever tell you mine?
sadlilgrrl (7:46:18 PM): your pic said dan
sadlilgrrl (7:46:20 PM): is that you?
arthinice (7:46:26 PM): yes
arthinice (7:46:35 PM): did you send me all the pix you have?
sadlilgrrl (7:46:43 PM): yaeh
sadlilgrrl (7:46:44 PM): 4
arthinice (7:46:53 PM): i wish you had 1,000
sadlilgrrl (7:46:58 PM): me 2
sadlilgrrl (7:47:02 PM): but thatd be a lot
sadlilgrrl (7:47:03 PM): lol
arthinice (7:47:13 PM): but not even close to enough
sadlilgrrl (7:47:20 PM): lol well you can see me in june
sadlilgrrl (7:47:29 PM): ill tell my mom i like it here
arthinice (7:47:40 PM): you'd stay just to see me?
sadlilgrrl (7:47:46 PM): i raelly like you
sadlilgrrl (7:47:49 PM): is that bad?
arthinice (7:47:53 PM): no
sadlilgrrl (7:47:58 PM): okay
sadlilgrrl (7:48:02 PM): i was worried you didnt like me too
arthinice (7:48:28 PM): kristen - i want to make something perfectly clear - so there is no doubt in your mind - ok?
sadlilgrrl (7:48:32 PM): okay
arthinice (7:48:37 PM): i like you!!!
sadlilgrrl (7:48:40 PM): yay!
sadlilgrrl (7:48:41 PM): (emoti)
arthinice (7:48:49 PM): should i say it again?
sadlilgrrl (7:48:52 PM): if you want to
sadlilgrrl (7:48:53 PM): lol
arthinice (7:48:55 PM): i like you!!!
arthinice (7:49:06 PM): and...
arthinice (7:49:12 PM): i want to be with you!!!
sadlilgrrl (7:49:17 PM): yay!
sadlilgrrl (7:49:20 PM): me too!
sadlilgrrl (7:49:21 PM): lol
arthinice (7:49:29 PM): should i say that again too?
arthinice (7:49:41 PM): i want to be with you!!!
sadlilgrrl (7:49:45 PM): yay!
sadlilgrrl (7:49:47 PM): (emoti)
arthinice (7:50:02 PM): let me see if i can make some of that a little more clear - ok?
sadlilgrrl (7:50:06 PM): okay
sadlilgrrl (7:50:07 PM): (emoti)
arthinice (7:50:12 PM): i want to taste your sweet kisses
arthinice (7:50:26 PM): i want to hug and cuddle with you
arthinice (7:50:44 PM): i want to softly touch you in soft places
arthinice (7:51:01 PM): i want to gently kiss and suck on your nipples ( More to come! )
arthinice (7:51:09 PM): are you taking notes??
sadlilgrrl (7:51:13 PM): lol yeah
arthinice (7:51:24 PM): i want to kiss your sexy tummy
arthinice (7:51:29 PM): and your sexy legs
arthinice (7:51:58 PM): i want to stroke my tongue up and down your pussy
arthinice (7:52:06 PM): and probe your pussy with my fingers
sadlilgrrl (7:52:20 PM): your making me blush!
arthinice (7:52:30 PM): i want you to "shake"
arthinice (7:53:02 PM): i want to be the first guy to put his cock inside your pussy ( What else? )
sadlilgrrl (7:53:26 PM): that does sound really nice
arthinice (7:53:29 PM): i want you to feel how a cock pumping in and out of you feels ( Oh, that. )
sadlilgrrl (7:53:33 PM): you arent just kidding me are you?
arthinice (7:53:41 PM): no, baby
sadlilgrrl (7:53:44 PM): okay
arthinice (7:53:54 PM): does all that sound good?
sadlilgrrl (7:53:57 PM): yeah
sadlilgrrl (7:54:02 PM): it really does
sadlilgrrl (7:54:20 PM): can i hear your voice?
arthinice (7:54:27 PM): yes if you want to
sadlilgrrl (7:54:38 PM): i do
sadlilgrrl (7:54:42 PM): i bet its really sexy
arthinice (7:55:02 PM): i'm not sure about that - you can tell me later if i'm still a winner
sadlilgrrl (7:55:07 PM): okay
sadlilgrrl (7:55:13 PM): i'll tell you
sadlilgrrl (7:55:14 PM): (emoti)
arthinice (7:55:16 PM): will you have to get offline to call?
sadlilgrrl (7:55:22 PM): no
sadlilgrrl (7:55:27 PM): but ill hacve to go to the other room
sadlilgrrl (7:55:32 PM): cuz the cordless doesnt reach in here
sadlilgrrl (7:55:33 PM): is taht ok?
arthinice (7:55:37 PM): yes
arthinice (7:55:39 PM): but
arthinice (7:55:52 PM): i want you to check something for me first - ok?
sadlilgrrl (7:55:55 PM): okay
arthinice (7:56:10 PM): above this box is the word File - see it?
sadlilgrrl (7:56:15 PM): uhhuh
arthinice (7:56:27 PM): clik on it and go to Privacy Options...
sadlilgrrl (7:56:41 PM): okay after i do that i cant type to you
arthinice (7:57:03 PM): ok - go down til you see Archive and select it
sadlilgrrl (7:57:16 PM): okay tell me what do to n i wont tpe back for a minute
arthinice (7:57:37 PM): is there a check mark in the Enable Archiving box? ( That's just disturbing. )
sadlilgrrl (7:57:41 PM): no
arthinice (7:57:49 PM): whew!!!
sadlilgrrl (7:57:51 PM): i clicked on cancel
arthinice (7:57:53 PM): good!!!
sadlilgrrl (7:57:59 PM): so i can type again
sadlilgrrl (7:58:01 PM): is that right?
arthinice (7:58:09 PM): we do NOT want a check mark in that bos
arthinice (7:58:10 PM): box
sadlilgrrl (7:58:19 PM): okay
sadlilgrrl (7:58:21 PM): there isnt one
arthinice (7:58:24 PM): know why?
sadlilgrrl (7:58:35 PM): cuz then it would stay?
arthinice (7:59:05 PM): then someone could come in and read what we have said - and i could get into a lot of trouble
sadlilgrrl (7:59:11 PM): oh
arthinice (7:59:12 PM): and we don't want that - ight?
sadlilgrrl (7:59:12 PM): okay
sadlilgrrl (7:59:13 PM): no
arthinice (7:59:19 PM): (emoti)
arthinice (7:59:30 PM): i'm trying to protect you and me - both ( That's what it sounds like. )
sadlilgrrl (7:59:37 PM): yeah i dont wanna get grounded.
arthinice (7:59:53 PM): i'm afraid it would be worse than grounding
sadlilgrrl (7:59:59 PM): oh
arthinice (8:00:21 PM): for me - it would be 25-30 yrs in jail ( We can hope! )
sadlilgrrl (8:00:25 PM): oh no!
arthinice (8:00:42 PM): so - we must be very careful - ok?
sadlilgrrl (8:00:46 PM): okay
arthinice (8:01:04 PM): be very careful about who you tell about this
sadlilgrrl (8:01:12 PM): i wont tell anyone
arthinice (8:01:19 PM): ok
arthinice (8:01:25 PM): am i scaring you?
sadlilgrrl (8:01:29 PM): no
sadlilgrrl (8:01:32 PM): why would you?
arthinice (8:01:42 PM): because it is a little scary
sadlilgrrl (8:01:47 PM): i trust you though
arthinice (8:01:48 PM): risky
arthinice (8:02:02 PM): cool
arthinice (8:02:23 PM): ok - one more thing before i hear your sweet voice - ok?
sadlilgrrl (8:02:28 PM): okay
arthinice (8:02:46 PM): i want to be honest with you - make sure there are no surprises
sadlilgrrl (8:02:55 PM): okay
arthinice (8:02:56 PM): we haven't talked about me very much
arthinice (8:03:14 PM): i think you should know that i am married
sadlilgrrl (8:03:25 PM): that's okay
arthinice (8:03:26 PM): and we have to be careful about that too
sadlilgrrl (8:03:30 PM): were just gonna practice, right?
arthinice (8:03:38 PM): uh huh
sadlilgrrl (8:03:41 PM): okay
arthinice (8:03:54 PM): are you cool with all that?
sadlilgrrl (8:04:00 PM): yeah
arthinice (8:04:10 PM): ready to call me?
sadlilgrrl (8:04:17 PM): yeah
arthinice (8:04:34 PM): 479 435-1424
sadlilgrrl (8:04:49 PM): okay ill go call now.
arthinice (8:04:53 PM): please close this box before you leave the room - ok? ( Whatever for, Dan? )
sadlilgrrl (8:04:57 PM): okay
sadlilgrrl (8:05:08 PM): closing it now n going to call
arthinice (8:05:13 PM): ok

Verifier: Ladybass. In her own words:
(I called Dan and he told me that I sounded more mature then 13 years old. Told me that if I weren't nervous about trying this stuff he would be concerned. I said I was nervous, told me that I was sweet and that I had a sweet voice. I basically told him that my phone was dying and that I would talk to him online. He said ok.)

arthinice (8:10:52 PM): wow wow wow wow!!!!!
sadlilgrrl (8:11:15 PM): what?
arthinice (8:11:33 PM): you are sooooo sexy!!!
sadlilgrrl (8:11:43 PM): (emoti)
arthinice (8:12:01 PM): you sound so mature
sadlilgrrl (8:12:04 PM): i -am-
sadlilgrrl (8:12:07 PM): (emoti)
arthinice (8:12:12 PM): you sound much older than 14
arthinice (8:12:16 PM): more like 18 or so ( She doesn't. I've heard her. )
sadlilgrrl (8:12:18 PM): tyvm!
arthinice (8:12:45 PM): can i tell you another little secret?
sadlilgrrl (8:12:49 PM): yeah
arthinice (8:12:52 PM): i like you!!!
sadlilgrrl (8:13:18 PM): i ilke you tooo!
sadlilgrrl (8:13:25 PM): you have such a distingushed voice
arthinice (8:13:32 PM): that's probably not a secret anymore - huh
sadlilgrrl (8:13:41 PM): to everyone but me!
arthinice (8:13:47 PM): (emoti)
arthinice (8:14:10 PM): and you sure know how to pump my ego too
sadlilgrrl (8:14:21 PM): i mean it
arthinice (8:14:42 PM): i know - and that's what makes it that much more special
sadlilgrrl (8:14:51 PM): (emoti)
arthinice (8:14:57 PM): you are a very special girl
sadlilgrrl (8:15:02 PM): thank you
sadlilgrrl (8:15:03 PM): ""> (messed up blushing emoti)
sadlilgrrl (8:15:05 PM): (emoti)
arthinice (8:15:35 PM): i'm soo glad you found me
sadlilgrrl (8:15:41 PM): i'm glad you found me!
sadlilgrrl (8:15:46 PM): what made you msesag me?
sadlilgrrl (8:15:48 PM): message?
arthinice (8:16:04 PM): lol - are you sure you want me to tell you?
sadlilgrrl (8:16:07 PM): yeah
arthinice (8:16:13 PM): i may get myself in trouble
sadlilgrrl (8:16:19 PM): thats okay
arthinice (8:16:45 PM): honestly - i have been looking for a young girl
arthinice (8:16:57 PM): i really didn't think i would find one as young as you
sadlilgrrl (8:17:07 PM): am i okay?
arthinice (8:17:14 PM): in fact - i haven't been able to find any at all
arthinice (8:17:36 PM): are you okay?? baby - you passed okay a loonnngggg lonnngggg time ago
arthinice (8:17:43 PM): you are much much more than okay!!!
sadlilgrrl (8:17:45 PM): im so glad
arthinice (8:17:57 PM): am i in trouble?
sadlilgrrl (8:18:08 PM): no cuz you picked me
sadlilgrrl (8:18:08 PM): lol
arthinice (8:18:14 PM): (emoti)
arthinice (8:18:44 PM): i have chatted with a lot lot lot of girls - lots of ages
arthinice (8:18:51 PM): but none as sweet as you
sadlilgrrl (8:18:57 PM): (emoti)
sadlilgrrl (8:19:04 PM): when are you gonna be in lr?
arthinice (8:19:10 PM): some of them really like to play stupid mind games with me
sadlilgrrl (8:19:16 PM): why?!
arthinice (8:19:30 PM): i wish i could figure that one out myself
sadlilgrrl (8:19:48 PM): thats so mean
arthinice (8:20:06 PM): they send me pix and tell me they want to meet and make me feel really good - but...
arthinice (8:20:18 PM): they disappear - poof!!!
arthinice (8:20:26 PM): and i never hear from them again
sadlilgrrl (8:20:30 PM): oh
sadlilgrrl (8:20:32 PM): im so sorry
arthinice (8:20:43 PM): but you are different
sadlilgrrl (8:20:57 PM): yes i am
arthinice (8:21:04 PM): none of them would let me call or they didn't want to call me
sadlilgrrl (8:21:11 PM): thats so wrong
arthinice (8:21:17 PM): you did
arthinice (8:21:28 PM): why am i telling you all this??
sadlilgrrl (8:21:40 PM): lol cuz im a good listener too?
arthinice (8:21:47 PM): yes - you are!!
arthinice (8:21:59 PM): but you don't need to hear all this
sadlilgrrl (8:22:15 PM): i dont mind
arthinice (8:22:29 PM): you are different - and i like you
sadlilgrrl (8:22:33 PM): i like you too
arthinice (8:22:50 PM): will you do me one tiny favor before we chat again next time?
sadlilgrrl (8:22:54 PM): okay
arthinice (8:23:02 PM): pick a new nick??
sadlilgrrl (8:23:05 PM): lol
sadlilgrrl (8:23:05 PM): okay
sadlilgrrl (8:23:09 PM): happylilgrrl?
sadlilgrrl (8:23:10 PM): lol
arthinice (8:23:18 PM): that sounds good
sadlilgrrl (8:23:29 PM): i have to go get dinner ready before gramma comes home
sadlilgrrl (8:23:32 PM): will you be on tomorrow night?
arthinice (8:23:48 PM): yes - from about 5:00 til about 6:30 or so
sadlilgrrl (8:23:53 PM): okay ill try to be on then
arthinice (8:24:00 PM): is that a good time for you?
sadlilgrrl (8:24:06 PM): yeah
arthinice (8:24:16 PM): i need to try to not chat much later than that if i can
sadlilgrrl (8:24:24 PM): okay
sadlilgrrl (8:24:27 PM): ill be on then
arthinice (8:24:43 PM): but - if i'm chatting with you - time will quickly slip away
sadlilgrrl (8:24:55 PM): (emoti)
arthinice (8:25:16 PM): we have already chatted 2 1/2 hr since you went to fix dinner the first time
sadlilgrrl (8:25:25 PM): yeah i have to finish the stuff now
sadlilgrrl (8:25:26 PM): lol
arthinice (8:25:32 PM): and i need to go too
arthinice (8:25:37 PM): but i don't want to
sadlilgrrl (8:25:42 PM): yeah i know the feeling
arthinice (8:25:51 PM): i like you, baby
arthinice (8:25:56 PM): i'll see you tomorrow
sadlilgrrl (8:26:20 PM): i like you too
sadlilgrrl (8:26:21 PM): n okay!
sadlilgrrl (8:26:23 PM): (emoti)
arthinice (8:26:26 PM): night
sadlilgrrl (8:26:26 PM): (emoti)
sadlilgrrl (8:26:29 PM): night!
arthinice (5:08:12 PM): kristen?
sadlilgrrl (5:08:18 PM): hey!!
arthinice (5:08:23 PM): hey, sweetie
arthinice (5:08:28 PM): sure have missed you!!!
sadlilgrrl (5:08:37 PM): yeah i had major work to do and had to clean
sadlilgrrl (5:08:39 PM): lol
arthinice (5:09:00 PM): have you been a good girl?
sadlilgrrl (5:09:12 PM): more or less
sadlilgrrl (5:09:13 PM): (emoti)
arthinice (5:09:21 PM): what have you been doing??
sadlilgrrl (5:09:35 PM): nothing really
sadlilgrrl (5:09:43 PM): i stayed up really late last night n watched the goonies
arthinice (5:10:09 PM): the goonies?
sadlilgrrl (5:10:16 PM): yah its a movie
arthinice (5:10:21 PM): was it good?
sadlilgrrl (5:10:33 PM): it was funny
arthinice (5:10:45 PM):
sadlilgrrl (5:10:51 PM): how have YOU been?
arthinice (5:11:02 PM): besides missing you - i've been ok
arthinice (5:11:17 PM): i have thought about you an awful lot the last few days
sadlilgrrl (5:11:29 PM): im glad!
sadlilgrrl (5:11:32 PM): ive thought about you too
sadlilgrrl (5:11:34 PM):
sadlilgrrl (5:15:02 PM): did you disappear?
sadlilgrrl (5:15:20 PM): aww ill be back later
arthinice (5:56:46 PM): i'm sorry - i'm still at work and somebody came in - i hope you will be back online soon...
"""
lines = conversation2.strip().split('\n')

# Function to process a conversation and count entity types
def count_entity_types(conversation):
    # Process the conversation with spaCy NER
    doc = nlp(conversation)
    # Count occurrences of each entity type
    entity_counts = {}
    for ent in doc.ents:
        entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1
    return entity_counts

# Process and count entities for each conversation
entity_counts_conversation1 = count_entity_types(conversation1)
entity_counts_conversation2 = count_entity_types(conversation2)

# Display the summary of entity counts
print("Entity counts for Conversation 1:", entity_counts_conversation1)
print("Entity counts for Conversation 2:", entity_counts_conversation2)

from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
# Initialize counters
positive_count = 0
negative_count = 0

for line in lines:
    result = sentiment_pipeline(line)[0]
    # Update counters based on sentiment
    if result['label'] == 'POSITIVE':
        positive_count += 1
    elif result['label'] == 'NEGATIVE':
        negative_count += 1
    # print(f"Line: {line}\nSentiment: {result['label']}, Score: {result['score']}\n")
    
# After processing all lines
print(f"Positive Sentiments: {positive_count}, Negative Sentiments: {negative_count}")


if positive_count > negative_count:
    print("The conversation has more positive sentiment.")
elif negative_count > positive_count:
    print("The conversation has more negative sentiment.")
else:
    print("The conversation has an equal amount of positive and negative sentiments.")
doc = nlp(conversation)

for ent in doc.ents:
    print(ent.text, ent.label_)
    
doc = nlp(conversation)
entities = [(ent.text, ent.label_) for ent in doc.ents]

personal_info_questions = 0
relate_and_build_trust = 0
isolation_indicators = 0

# keywords extracted using the CHATGPT suggestions
personal_info_keywords = ['where do you live', 'how old are you', 'what school', 'what grade', 'do u go to school', 'where is ur mom', 'u got a pic']
relate_build_trust_keywords = ['me too', 'i also', 'i feel', 'i understand',  'had a similar experience', 'in this together', 'understand how you feel','not easy','here for you']
isolation_keywords = ['alone', 'by yourself', 'only child', 'divorced', 'talk to anyone', 'on my own', 'isolated from everyone']

for sentence in doc.sents:
    sentence_text = sentence.text.lower()
    if any(keyword in sentence_text for keyword in personal_info_keywords):
        personal_info_questions += 1
    if any(keyword in sentence_text for keyword in relate_build_trust_keywords):
        relate_and_build_trust += 1
    if any(keyword in sentence_text for keyword in isolation_keywords):
        isolation_indicators += 1

features = {
    'personal_info_questions': personal_info_questions,
    'relate_and_build_trust': relate_and_build_trust,
    'isolation_indicators': isolation_indicators,
}

print(features)