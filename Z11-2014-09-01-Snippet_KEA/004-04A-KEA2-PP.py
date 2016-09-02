# -*- coding: utf-8 -*-

#
#################################################
import sys
import os
import re
import nltk
import time
import numpy #needed for nltk named entity recognition
import string

#################################################
# this script takes about 2,3 min for 13k enron emails

print("USAGE: python 004-04A-KEA2-PP.py inputfolder outputfolder")
print("EXAMPLE: python 004-04A-KEA2-PP.py data02-KEA2-train/ data02-KEA2-PP-train")

if (len(sys.argv)<3):
  print("Not enough arguments")
  exit()

#################################################

### nltk supports some tagged corpus already
# check 
#   dir(nltk.corpus)
#   dir(nltk.corpus.brown)
#   nltk.corpus.brown.categories()
#   nltk.corpus.brown.tagged_sents(categories=['reviews'])

# use 'government', 'hobbies', 'humor', 'learned', 'news', 'reviews',
tZ02_tsen_train=nltk.corpus.brown.tagged_sents(categories=[
  'government', 'hobbies', 'humor', 'learned', 'news', 'reviews'])

###
def tZ02_backoff_tagger(tagged_sents, tagger_classes, backoff=None):
  if not backoff:
    backoff = tagger_classes[0](tagged_sents)
    del tagger_classes[0]
  
  for cls in tagger_classes:
    tagger = cls(tagged_sents, backoff=backoff)
    backoff = tagger
  
  return backoff


tZ02_word_patterns = [
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
    (r'.*ould$', 'MD'),
    (r'.*ing$', 'VBG'),
    (r'.*ed$', 'VBD'),
    (r'.*ness$', 'NN'),
    (r'.*ment$', 'NN'),
    (r'.*ful$', 'JJ'),
    (r'.*ious$', 'JJ'),
    (r'.*ble$', 'JJ'),
    (r'.*ic$', 'JJ'),
    (r'.*ive$', 'JJ'),
    (r'.*ic$', 'JJ'),
    (r'.*est$', 'JJ'),
    (r'^a$', 'PREP'),
]

tZ02_raubt_tagger = tZ02_backoff_tagger(tZ02_tsen_train, 
  [nltk.tag.AffixTagger, nltk.tag.UnigramTagger, 
    nltk.tag.BigramTagger, nltk.tag.TrigramTagger],
  backoff=nltk.tag.RegexpTagger(tZ02_word_patterns))

#tZ02_raubt_tagger.tag(nltk.word_tokenize('where   is   the   location     of'))

##########

tZ01_regc01_link=re.compile(r"(?P<pf>(^|\W))([a-zA-Z]+://|www\.)\w[\w.]*\w",
  re.IGNORECASE)
#tZ01_regc01_link.sub("\g<pf> ","http://fasd.com www.com.asdf www.sdfa$ wwwacom.sdf")

tZ01_regc02_email=re.compile(r"(?P<pf>(^|\W))\w[\w.]+\w@\w[\w.]+\w")
#tZ01_regc02_email.sub("\g<pf> ","abc@gmail.com asfd.asdf@com. asdf$sf@gmail.com sf.@dsf")

tZ01_regc03_file=re.compile(r"""
    (?P<pf>(^|\W))
    [\w()-_\[\]@~<>.]+
    \.
    (com|txt|doc[x]?|xls[s]?|exe|zip|ppt[x]?|
      rar|img|gif|png|bmp|jp[e]?g|htm[l]?|url)
  """,re.VERBOSE|re.IGNORECASE)
#tZ01_regc03_file.sub("\g<pf> ","asf.doc sdf.exe sadf.doc sadf2.doc sdfa.xlss")

tZ01_regc04_num=re.compile(r"(?P<pf>(^|\W))[-+ ]*\d[\d.',]*(\d[eE]?[-+ ]*\d[\d.',]+)?")
#tZ01_regc04_num.sub("\g<pf> ","-+- 123.23.'123e -123.123 daf")

tZ01_regc05_punct=re.compile(r"(\t|\n)+")

#remove punctuations; keep ";'.!?"
tZ01_regc05_punct1=re.compile(r'[\"`£$%^&\*:@#~<>,/\\\|_+=:]+')
#tZ01_regc05_punct.sub(" ","i won't 123£ `! $%^&* -_+= dfasdf |{ asfs dfassd fa}. sdsaf; dsf: a \\ / , 123")

#add space between characters and punctuations (except ')
tZ01_regc05_punct2=re.compile(r"(?P<pf>\w)(?P<pu>[;.!?]+)")
#tZ01_regc05_punct2.sub("\g<pf> \g<pu>","afdf. sdfas sdf.sdfas")

tZ01_regc05_punct3=re.compile(r"(?P<pu>[;.!?]+)(?P<sf>\w)")
#tZ01_regc05_punct3.sub("\g<pu> \g<sf>","afdf. sdfas sdf.sdfas")

#only keep noun and verbs for LDA
tZ01_regc20_pos1=re.compile(r"^(N[NP]|VB|PP|\.)",re.IGNORECASE)



###
# keep "email", "emails", "mail", "mails", "phone","phones" for KEA
tZ01_stopwords=nltk.corpus.stopwords.words("english") +  [
  "cc","bcc","po","tel","fax","am","pm","n't","etc",
    "'ve","'ll",
    "thanks", "thank", "tkx", "thx", "sorry", "regards",
    "luck", "wish", "wishes", "hello", "bye", "disposal",
    "forward","forwards","forwarded",
    "dear", "hi", "fyi",
    "let", "lets",              #this may be useful for action items but not topics
    "get", "gets", "got",
    "please", "pls","need","needs",
    "average","sum",
    #
    "monday","mon","tuesday","tue","wednesday","thursday","thu","friday","fri",
    "saturday","sat","sunday",
    "january","jan","february","feb",
    "march","mar",  #march?
    "april","apr",
    #"may", #may is already a stopword
    "june","jun","july","jul","august","aug","september","sep","october","oct",
    "november","nov","december","dec",
    #
    "second","seconds","minute","minutes","hour","hours","day","days",
    "week","weeks","month","months","year","years",
    #
    "total","all",
    "kwh","mw",
    #
    "src","img","gif","div","bgcolor","html","serif","href","font","nbsp",
    "sansserif",
    # not sure about the below parts
    "know", "knows", "knew", 
    "understand", "understands",
    "doubt", "doubts",
    #"email", "emails", "mail", "mails", "phone","phones", 
    "web",
    #company names
    "enron"
    ]

### keeps sentence breaker (.;!?) and capital words.
def tZ01_regexp01(str01):
  str02=tZ01_regc01_link.sub("\g<pf> ",str01)
  str02=tZ01_regc02_email.sub("\g<pf> ",str02)
  str02=tZ01_regc03_file.sub("\g<pf> ",str02)
  str02=tZ01_regc04_num.sub("\g<pf> ",str02)
  str02=tZ01_regc05_punct.sub(" ",str02)
  str02=tZ01_regc05_punct1.sub(" ",str02)
  str02=tZ01_regc05_punct2.sub("\g<pf> \g<pu>",str02)
  str02=tZ01_regc05_punct3.sub("\g<pu> \g<sf>",str02)
  #str03 = " ".join([w for w in str02.split() if (not w in tZ01_stopwords)])
  str03=nltk.sent_tokenize(str02)
  
  str06=[]
  str07=""
  for str04 in str03:
    str05=tZ02_raubt_tagger.tag(nltk.word_tokenize(str04))
    #str05=nltk.pos_tag(nltk.word_tokenize(str04))
    #print(str05)
    for j in range(len(str05)):
      pos=str05[j][1]
      word=str05[j][0]
      if ((pos==".") or 
        (((pos==None) or (tZ01_regc20_pos1.match(pos))) 
          and (not word.lower() in tZ01_stopwords) 
          and (len(word)>2)
          and (len(word)<20)) ):
        
        str06.append(word)
    str07=" ".join(str06)
  
  return str07



# remove email header (oneline and multiline)
tZ01_regc34_emheader01=re.compile(r"""
  ([ \t>]*)
  (To:[ \t]|Cc:[ \t]|Bcc:[ \t]|Received:[ \t])
  ((.|\n)*?)
  (?=
    (Content-Class:[ \t]|MIME-Version:[ \t]|Date:[ \t]|
      From:[ \t]|Sent:[ \t]|To:[ \t]|Subject:[ \t]|\[mailto:[ \t]?|
      Thread-Topic:[ \t]|Thread-Index:[ \t]|Cc:[ \t]|Bcc:[ \t]|
      Full-name:[ \t]|Message-ID:[ \t]|Return-Path:[ \t]|Received:[ \t]|
      Content-Transfer-Encoding:[ \t]|
      X-MimeOLE:[ \t]|X-MS-Has-Attach:[ \t]|X-MS-TNEF-Correlator:[ \t]|
      X-OriginalArrivalTime:[ \t]|X-MIME-Autoconverted:[ \t]|X-UIDL:[ \t]|
      X-Keywords:[ \t]|X-Mailer:[ \t]|Hello[\W]|Hi[\W]|Dear[\W])
  )
  """,re.VERBOSE|re.IGNORECASE)
tZ01_regc34_emheader02=re.compile(r"""
  ([ \t>]*)
  (Content-Class:[ \t]|MIME-Version:[ \t]|Date:[ \t]|
    From:[ \t]|Sent:[ \t]|To:[ \t]|Subject:[ \t]|\[mailto:[ \t]?|
    Thread-Topic:[ \t]|Thread-Index:[ \t]|Cc:[ \t]|Bcc:[ \t]|
    Full-name:[ \t]|Message-ID:[ \t]|Return-Path:[ \t]|Received:[ \t]|
    Content-Transfer-Encoding:[ \t]|
    X-MimeOLE:[ \t]|X-MS-Has-Attach:[ \t]|X-MS-TNEF-Correlator:[ \t]|
    X-OriginalArrivalTime:[ \t]|X-MIME-Autoconverted:[ \t]|X-UIDL:[ \t]|
    X-Keywords:[ \t]|X-Mailer:[ \t])
  ((.|\n){0,200}?)
  (?=
    (Content-Class:[ \t]|MIME-Version:[ \t]|Date:[ \t]|
      From:[ \t]|Sent:[ \t]|To:[ \t]|Subject:[ \t]|\[mailto:[ \t]?|
      Thread-Topic:[ \t]|Thread-Index:[ \t]|Cc:[ \t]|Bcc:[ \t]|
      Full-name:[ \t]|Message-ID:[ \t]|Return-Path:[ \t]|Received:[ \t]|
      Content-Transfer-Encoding:[ \t]|
      X-MimeOLE:[ \t]|X-MS-Has-Attach:[ \t]|X-MS-TNEF-Correlator:[ \t]|
      X-OriginalArrivalTime:[ \t]|X-MIME-Autoconverted:[ \t]|X-UIDL:[ \t]|
      X-Keywords:[ \t]|X-Mailer:[ \t]|Hello[\W]|Hi[\W]|Dear[\W])
  )
  """,re.VERBOSE|re.IGNORECASE)

#remove Original Message
tZ01_regc35_OM=re.compile(r"""
  -+\s*Original\s+Message\s*-+
  """,re.VERBOSE|re.IGNORECASE)


#remove enron disclaimer
tZ01_regc32_dcl=re.compile(r"([>*]|\s)+This\s+e-mail\s+is\s+the\s+property(.|\n)*estoppel\s+or\s+otherwise\.\s+Thank you\.([>*]|\s)+",re.IGNORECASE)
#print(tZ01_regc32_dcl.sub("\n",t1_str))

#remove block
tZ01_regc33_sep=re.compile(r"""
  (?P<pf>(?P<pfc>[+*_=#~<>/-])(?P=pfc){9,})
  (
    ([> \t\n\r\f\v]*)([^\s+*_=#~<>/-]+)
    (([> \t\n\r\f\v]+[^>\s]+){0,300}?)
    ([> \t\n\r\f\v]*)
  )
  (?P=pf)
  """,re.VERBOSE|re.IGNORECASE)

#################################################


tA01_dir1 = sys.argv[1]
tA01_dir2 = sys.argv[2]

#tA01_dir1 = "data02-test"
#tA01_dir2 = "data02-clean01"

print("source folder: ",tA01_dir1,"| output folder: ", tA01_dir2)


time1=time.time()
for t1_root, t1_sfolders, t1_fns in os.walk(tA01_dir1,followlinks=True):
  t1_dir4=os.path.join(tA01_dir2,os.path.relpath(t1_root,tA01_dir1))
  if not os.path.exists(t1_dir4):
    os.makedirs(t1_dir4)
  for t1_fn0 in t1_fns: 
    t1_fn1=os.path.join(t1_root,t1_fn0)
    t1_fn2=os.path.join(t1_dir4,t1_fn0)
    t1_file=open(t1_fn1,"r")
    t1_file2=open(t1_fn2,"w")
    t1_str=filter(lambda x: x in string.printable,t1_file.read())
    t1_str02=tZ01_regc34_emheader01.sub(r""" """,t1_str)
    t1_str02=tZ01_regc34_emheader02.sub(r""" """,t1_str02)
    t1_str03=tZ01_regc32_dcl.sub("\n",t1_str02)
    t1_str04=tZ01_regc35_OM.sub(" ",t1_str03)
    t1_str05=tZ01_regc33_sep.sub("\n",t1_str04)
    t1_str06=tZ01_regexp01(t1_str05)
    #stop_here
    t1_file2.write(t1_str06)
    #print(t1_str02)
    t1_file.close()
    t1_file2.close()

time2=time.time()
print("Total running time: ",time2-time1)








