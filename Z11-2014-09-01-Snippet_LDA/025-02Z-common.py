# -*- coding: utf-8 -*-
#################################################


tY10_tsen_train=nltk.corpus.brown.tagged_sents(categories=[
  'government', 'hobbies', 'humor', 'learned', 'news', 'reviews'])

###
def tY10_backoff_tagger(tagged_sents, tagger_classes, backoff=None):
  if not backoff:
    backoff = tagger_classes[0](tagged_sents)
    del tagger_classes[0]
  
  for cls in tagger_classes:
    tagger = cls(tagged_sents, backoff=backoff)
    backoff = tagger
  
  return backoff


tY10_word_patterns = [
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
    (r'.*ould$', 'MD'),
    #(r'.*ing$', 'VBG'),
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

tY10_raubt_tagger = tY10_backoff_tagger(tY10_tsen_train, 
  [nltk.tag.AffixTagger, nltk.tag.UnigramTagger, 
    nltk.tag.BigramTagger, nltk.tag.TrigramTagger],
  backoff=nltk.tag.RegexpTagger(tY10_word_patterns))

#tY10_raubt_tagger.tag(nltk.word_tokenize('where   is   the   location     of'))

#################################################

tY11_NPChunk=nltk.RegexpParser(r"""
  P1JN: { <JJ.*>+(<N.*>+<Z1>?<N.*>+|<N.*>+) }
  P2N: { <N.*>+ }""")

#Lemmatizer
tY12_Lem=nltk.stem.wordnet.WordNetLemmatizer()
def tY12_lemmatize(word):
  res=tY12_Lem.lemmatize(word,"v")
  wordn=tY12_Lem.lemmatize(word,"n")
  if (len(wordn)<len(res)):
    res=wordn
  return res

#
tY13_stopwords=nltk.corpus.stopwords.words("english") +  [
  "cc","bcc","po","tel","fax","am","pm","n't","etc",
    "luck", "wish", "wishes", "hello", "bye", "disposal",
    "thanks","fyi","regards","sorry","wish",
    #
    "date",
    "monday","mon","tuesdafy","tue","wednesday","thursday","thu","friday","fri",
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
    #
    "src","img","gif","div","bgcolor","html","serif","href","font","nbsp"]


tY13_stopwords_lem=[tY12_lemmatize(w) for w in tY13_stopwords]

#######################

tY01_regc00C_startW =re.compile(r"^\W")
#tY01_regc00C_startW.match(r'"")

tY01_regc00D_uppercase =re.compile(r"^[A-Z0-9\s\W]+$")
#tY01_regc00D_uppercase.match("AYF ASHGD  £$ ASD  $ 67")

tY01_regc01A_num=re.compile(r"(?P<pf>(^|\W))[-+ ]*\d[\d.',]*(\d[eE]?[-+ ]*\d[\d.',]+)?")
#tY01_regc01_num.sub("\g<pf> ","-+- 123.23.'123e -123.123 daf 76% 76 76.7 fgas")



tY01_regc02A_linebreak=re.compile(r"(\r?\n)+")
#tY01_regc02_linebreak.sub(".","&&&&\r\nHe is presently")

#remove punctuations; keep ";.!?,:|-"
tY01_regc02B_rmpunct=re.compile(r"""[\"`£$%^&\*@#~<>/\\\_+=']+""")
#tY01_regc02B_rmpunct.sub(" ",r"""i won't 123£ `! $%^&* -_+=" dfa"s:df |{ asfs dfassd fa}. sdsaf; dsf: a \\ / , 123""")

#add space between characters and punctuations
tY01_regc02C_punctaddls=re.compile(r"(?P<pf>\w)(?P<pu>[;.!?,:|-]+)")
#tY01_regc02C_punctaddls.sub("\g<pf> \g<pu>","afdf. sdfas sdf.sdfas")

tY01_regc02D_punctaddrs=re.compile(r"(?P<pu>[;.!?,:|-]+)(?P<sf>\w)")
#tY01_regc02D_punctaddrs.sub("\g<pu> \g<sf>","afdf. sdfas sdf.sdfas")

tY01_regc02E_rmspace1=re.compile(r"\s{2,}")
#tY01_regc02E_rmspace1.sub(" ","afdf. sdfas sdf.  \nsdfas")

tY01_regc02F_rmspace2=re.compile(r"^\s+")
#tY01_regc02F_rmspace2.sub("","   afdf. sdfas sdf.  \nsdfas")

#remove left spaces of posession quotes
tY01_regc07A_posquote =re.compile(r"[ ]+(?P<sf>['])")

#
tY01_regc09A_abbreviation =re.compile(r"^([A-Z0-9]{3,}[A-Z0-9_-]*|[A-Z0-9_-]*[A-Z0-9]{3,})$")
#tY01_regc09A_abbreviation.match("DQPSK")
#################################################

