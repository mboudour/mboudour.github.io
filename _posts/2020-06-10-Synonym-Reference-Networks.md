
# <span style="color:green">Network of synonyms from NLTK WordNet</span>
## <span style="color:blue">Moses Boudourides</span>


```python
import math, random, pickle, collections, operator, string, community 
import itertools as it
import pandas as pd
import numpy as np
import networkx as nx
from networkx import NetworkXNoPath
from networkx.drawing.nx_agraph import graphviz_layout
import pygraphviz
import matplotlib.pyplot as plt
import matplotlib as mpl
from nltk.corpus import wordnet as wn
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=RuntimeWarning) 
```


```python
def g_diagnostics(G,st):
    
    print("The %s has %i nodes and %i edges \n" %(st,len(G.nodes()), len(G.edges())))

    if G.is_directed()==True:
        print("The %s is a directed graph" %st)
    else:
        print("The %s is an undirected graph" %st)    
    if nx.is_weighted(G)==True:
        print("The %s graph is a weighted graph" %st)
    else:
        print("The %s graph is an unweighted graph" %st)
    if G.is_directed()==False:
        if nx.is_connected(G)==True:
            print("The %s is a connected graph" %st)
        else:
            print("The %s graph is a disconnected graph and it has %i connected components" %(st,nx.number_connected_components(G)))   
            giant = max(nx.connected_components(G), key=len)
            Glcc=G.subgraph(giant)
            print("The largest connected component of this graph has %i nodes and %i edges" %(len(Glcc.nodes()),len(Glcc.edges())))
    else:
        if nx.is_strongly_connected(G)==True:
            print("The %s is a strongly connected graph" %st)
        else:
            print("The %s graph is not strongly connected and it has %i strongly connected components" %(st,nx.number_strongly_connected_components(G)))
            giant = max(nx.strongly_connected_components(G), key=len)
            Glcc=G.subgraph(giant)
            print("The largest strongly connected component of this graph has %i nodes and %i edges" %(len(Glcc.nodes()),len(Glcc.edges())))
        if nx.is_weakly_connected(G)==True:
            print("The %s is a weakly connected graph" %st)
        else:
            print("The %s graph is not weakly connected and it has %i weakly connected components" %(st,nx.number_weakly_connected_components(G)))
            giantw = max(nx.weakly_connected_components(G), key=len)
            Glwcc=G.subgraph(giantw)
            print("The largest weakly connected component of this graph has %i nodes and %i edges" %(len(Glwcc.nodes()),len(Glwcc.edges())))
    print("The density of the %s is %.3f" %(st,nx.density(G)))   
    print("The transitivity of the %s is %.3f" %(st,nx.transitivity(G))) 
    if G.is_directed()==True:
        print("The reciprocity of the %s is %.3f" %(st,nx.reciprocity(G)))

def g_diameter(G,st):    
    try:
        diameter=nx.diameter(G)
        print("The diameter of the largest strongly connected component of the %s is %i" %(st,diameter))
    except Exception as e: 
        print(e)
```


```python
def syn_ant(word):
    synonyms = []
    isynonyms=[]
    antonyms = []
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    synonyms = sorted(set([w for w in synonyms if w!=word and w.lower()!=word and word not in w and w not in word]))
    synonyms = sorted(set([w for w in synonyms if "ise" not in w and "isation" not in w]))
    antonyms = sorted(set([w for w in antonyms if w!=word and w.lower()!=word and word not in w]))
    return (synonyms,antonyms) #isynonyms,
```

### The WordNet is lexical database in NLTK

#### There exist 117659 words in the WordNet database. However, some of them have empty sets of synonyms or antonyms.


```python
word = "good"
sa=syn_ant(word)
print("The synonyms of '%s' are: \n %s \n" %(word,sa[0]))
print("The antonyms of '%s' are: \n %s" %(word,sa[1]))
```

    The synonyms of 'good' are: 
     ['adept', 'beneficial', 'commodity', 'dear', 'dependable', 'effective', 'estimable', 'expert', 'full', 'honest', 'honorable', 'in_effect', 'in_force', 'just', 'near', 'practiced', 'proficient', 'respectable', 'right', 'ripe', 'safe', 'salutary', 'secure', 'serious', 'skilful', 'skillful', 'sound', 'soundly', 'thoroughly', 'undecomposed', 'unspoiled', 'unspoilt', 'upright', 'well'] 
    
    The antonyms of 'good' are: 
     ['bad', 'badness', 'evil', 'evilness', 'ill']



```python
# https://www.nltk.org/howto/wordnet.html

# synset1.path_similarity(synset2): 
# Return a score denoting how similar two word senses are, based on the shortest path that 
# connects the senses in the is-a (hypernym/hypnoym) taxonomy.
# The score is in the range 0 to 1.

# synset1.lch_similarity(synset2): 
# Leacock-Chodorow Similarity: Return a score denoting how similar two word senses are, 
# based on the shortest path that connects the senses (as above) 
# and the maximum depth of the taxonomy in which the senses occur. The relationship is given 
# as -log(p/2d) where p is the shortest path length and d the taxonomy depth.

# synset1.wup_similarity(synset2): 
# Wu-Palmer Similarity: Return a score denoting how 
# similar two word senses are, based on the depth of the two senses in the taxonomy and 
# that of their Least Common Subsumer (most specific ancestor node). 

w1='good'
w2='dear'
w3='bad'
w1s=wn.synsets(w1)[0]
w2s=wn.synsets(w2)[0]
w3s=wn.synsets(w3)[0]
print("The path similarity between %s and %s is %s" %(w1,w2,w1s.path_similarity(w2s)))
print("The path similarity between %s and %s is %s \n" %(w1,w3,w1s.path_similarity(w3s)))
print("The lch similarity between %s and %s is %s" %(w1,w2,w1s.lch_similarity(w2s)))
print("The lch similarity between %s and %s is %s \n" %(w1,w3,w1s.lch_similarity(w3s)))
print("The wup similarity between %s and %s is %s" %(w1,w2,w1s.wup_similarity(w2s)))
print("The wup similarity between %s and %s is %s" %(w1,w3,w1s.wup_similarity(w3s)))
```

    The path similarity between good and dear is 0.08333333333333333
    The path similarity between good and bad is 0.2 
    
    The lch similarity between good and dear is 1.1526795099383855
    The lch similarity between good and bad is 2.0281482472922856 
    
    The wup similarity between good and dear is 0.15384615384615385
    The wup similarity between good and bad is 0.6666666666666666


### Graph of References of Words to Synonyms


```python
nos=1500  #number of words

wns=list(wn.all_synsets()) 
syn_d={}
for w in random.sample(wns,nos): #wns: #
    w=w.lemmas()[0].name()
    sl=syn_ant(w)
    sl=syn_ant(w)[0]
    sl=list(set(sl))
    if len(sl)>0:
        syn_d[w]=sl

# for k,v in syn_d.items():
#     print("The synonyms of '%s' are %s \n" %(k,v))
```


```python
eds=[]
for k,v in syn_d.items():
    for vv in v:
        if vv in syn_d.keys():
            ks=wn.synsets(k)[0]
            vvs=wn.synsets(vv)[0]
            simi=ks.path_similarity(vvs)
            if simi==None:
                asimi=0
            else:
                asimi=1-ks.path_similarity(vvs)
            eds.append((k,vv,asimi))
print(len(eds),len(set(eds)))
eds
```

    195 195





    [('nub', 'heart', 0.9333333333333333),
     ('sound', 'go', 0.9166666666666666),
     ('sound', 'heavy', 0.9230769230769231),
     ('sound', 'audio', 0.875),
     ('chip', 'crisp', 0.9166666666666666),
     ('transfer', 'shift', 0.9090909090909091),
     ('shell', 'husk', 0.9230769230769231),
     ('define', 'set', 0.875),
     ('covering', 'hide', 0.9166666666666666),
     ('approximate', 'gauge', 0.9333333333333333),
     ('return', 'recall', 0.9230769230769231),
     ('return', 'render', 0.9285714285714286),
     ('even', 'tied', 0),
     ('compass_plant', "prairie_bird's-foot_trefoil", 0.0),
     ('complete', 'accomplished', 0.5),
     ('shift', 'transfer', 0.9090909090909091),
     ('shift', 'break', 0.8333333333333334),
     ('shift', 'shimmy', 0.8333333333333334),
     ('shift', 'stir', 0.9230769230769231),
     ('plump', 'go', 0.9285714285714286),
     ('cut', 'turn_off', 0),
     ('cut', 'deletion', 0.9166666666666666),
     ('stand_by', 'stick', 0.9090909090909091),
     ('keep', 'support', 0.9090909090909091),
     ('erupt', 'combust', 0.75),
     ('erupt', 'break', 0.8888888888888888),
     ('joint', 'stick', 0.9090909090909091),
     ('heavy', 'clayey', 0),
     ('heavy', 'grave', 0.9285714285714286),
     ('heavy', 'sound', 0.9230769230769231),
     ('heavy', 'large', 0.9230769230769231),
     ('race', 'rush', 0.8888888888888888),
     ('race', 'run', 0.9090909090909091),
     ('circle', 'set', 0.9090909090909091),
     ('rush', 'race', 0.8888888888888888),
     ('lead', 'go', 0.9230769230769231),
     ('lead', 'guide', 0.9333333333333333),
     ('lead', 'run', 0.9333333333333333),
     ('lead', 'head', 0.9230769230769231),
     ('twist', 'pull', 0.875),
     ('twist', 'turn', 0.9166666666666666),
     ('put_out', 'smother', 0.9090909090909091),
     ('put_out', 'release', 0.9166666666666666),
     ('try', 'render', 0.9230769230769231),
     ('fetching', 'convey', 0.9090909090909091),
     ('accomplished', 'complete', 0.5),
     ('home', 'internal', 0),
     ('home', 'house', 0.9166666666666666),
     ('reading', 'study', 0.9230769230769231),
     ('tied', 'attach', 0.6666666666666667),
     ('tied', 'even', 0.9166666666666666),
     ('tied', 'draw', 0.9230769230769231),
     ('stick', 'joint', 0.9090909090909091),
     ('stick', 'stand_by', 0),
     ('suffer', 'support', 0.9166666666666666),
     ('chess', 'cheat', 0.8),
     ('field', 'study', 0.9411764705882353),
     ('support', 'back', 0.9166666666666666),
     ('support', 'back_up', 0),
     ('support', 'patronize', 0),
     ('support', 'keep', 0.9090909090909091),
     ('support', 'suffer', 0),
     ('wiring', 'cable', 0.9333333333333333),
     ('set', 'circle', 0.9090909090909091),
     ('set', 'define', 0),
     ('turn', 'go', 0.9230769230769231),
     ('turn', 'ferment', 0.8888888888888888),
     ('turn', 'twist', 0.9166666666666666),
     ('turn', 'release', 0.9285714285714286),
     ('turn', 'act', 0.9166666666666666),
     ('taps', 'strike', 0.9230769230769231),
     ('Judges', 'try', 0.9166666666666666),
     ('Judges', 'approximate', 0),
     ('Judges', 'gauge', 0.9411764705882353),
     ('roll', 'wave', 0.8),
     ('remember', 'recall', 0.9),
     ('clayey', 'heavy', 0),
     ('rear', 'back', 0.9090909090909091),
     ('study', 'survey', 0.0),
     ('study', 'field', 0.9411764705882353),
     ('husk', 'shell', 0.9230769230769231),
     ('shot', 'stroke', 0.875),
     ('unpredictability', 'volatility', 0.875),
     ('go', 'lead', 0.9230769230769231),
     ('go', 'plump', 0.9285714285714286),
     ('go', 'turn', 0.9230769230769231),
     ('go', 'run', 0.9411764705882353),
     ('go', 'break', 0.9166666666666666),
     ('go', 'sound', 0.9166666666666666),
     ('wave', 'roll', 0.8),
     ('contracted', 'cut', 0.9230769230769231),
     ('cheat', 'darnel', 0.0),
     ('cheat', 'chess', 0.8),
     ('break', 'erupt', 0),
     ('break', 'shift', 0.8333333333333334),
     ('break', 'go', 0.9166666666666666),
     ('back_up', 'support', 0.8888888888888888),
     ('composition', 'makeup', 0.9285714285714286),
     ('release', 'put_out', 0),
     ('release', 'turn', 0.9285714285714286),
     ('Bond', 'stick', 0.9285714285714286),
     ('Bond', 'attach', 0),
     ('Bond', 'adhesiveness', 0.9375),
     ('back', 'support', 0.9166666666666666),
     ('back', 'rear', 0.9090909090909091),
     ('point', 'detail', 0.9),
     ('point', 'guide', 0.9444444444444444),
     ('point', 'head', 0.9375),
     ('draw_out', 'pull', 0.9166666666666666),
     ('render', 'try', 0.9230769230769231),
     ('render', 'return', 0.9285714285714286),
     ('intension', 'connotation', 0.0),
     ('turn_off', 'cut', 0.9166666666666666),
     ('rip', 'pull', 0.9230769230769231),
     ('stir', 'shift', 0.9230769230769231),
     ('pull', 'twist', 0.875),
     ('pull', 'draw', 0.9285714285714286),
     ('pull', 'draw_out', 0),
     ('pull', 'rip', 0.9230769230769231),
     ('guide', 'lead', 0.9333333333333333),
     ('guide', 'usher', 0.9166666666666666),
     ('guide', 'point', 0.9444444444444444),
     ('guide', 'draw', 0.9285714285714286),
     ('guide', 'run', 0.9473684210526316),
     ('guide', 'head', 0.9230769230769231),
     ('streak', 'run', 0.9375),
     ('detail', 'point', 0.9),
     ('audio', 'sound_recording', 0.9285714285714286),
     ('audio', 'sound', 0.875),
     ('grave', 'heavy', 0.9285714285714286),
     ('heart', 'nub', 0.9333333333333333),
     ('voicing', 'sound', 0.9166666666666666),
     ('darnel', 'cheat', 0.0),
     ('makeup', 'composition', 0.9285714285714286),
     ('survey', 'study', 0.0),
     ('house', 'home', 0.9166666666666666),
     ('usher', 'doorkeeper', 0.9090909090909091),
     ('usher', 'guide', 0.9166666666666666),
     ('raising', 'produce', 0.9285714285714286),
     ('raising', 'rear', 0.9166666666666666),
     ('raising', 'stir', 0.9230769230769231),
     ('shed', 'throw', 0.9333333333333333),
     ('melt', 'run', 0.9375),
     ('recall', 'return', 0.9230769230769231),
     ('recall', 'remember', 0),
     ('volatility', 'unpredictability', 0.875),
     ('sound_recording', 'audio', 0.9285714285714286),
     ('discretion', 'delicacy', 0.875),
     ('throw', 'stroke', 0.875),
     ('throw', 'shed', 0.9333333333333333),
     ('combust', 'erupt', 0.75),
     ('sharp', 'crisp', 0.9375),
     ('delicacy', 'discretion', 0.875),
     ('minor', 'child', 0.0),
     ('gauge', 'approximate', 0),
     ('crisp', 'sharp', 0.9375),
     ('crisp', 'chip', 0.9166666666666666),
     ('child', 'minor', 0.0),
     ('draw', 'pull', 0.9285714285714286),
     ('draw', 'guide', 0.9285714285714286),
     ('draw', 'run', 0.9444444444444444),
     ('seeking', 'try', 0.75),
     ('stroke', 'throw', 0.875),
     ('stroke', 'shot', 0.875),
     ('refinement', 'purification', 0.9285714285714286),
     ('refinement', 'cultivation', 0.9090909090909091),
     ('cad', 'heel', 0.9333333333333333),
     ('cultivation', 'refinement', 0.9090909090909091),
     ('deletion', 'cut', 0.9166666666666666),
     ('ferment', 'turn', 0.8888888888888888),
     ('large', 'heavy', 0.9230769230769231),
     ('run', 'lead', 0.9333333333333333),
     ('run', 'go', 0.9411764705882353),
     ('run', 'guide', 0.9473684210526316),
     ('run', 'race', 0.9090909090909091),
     ('run', 'draw', 0.9444444444444444),
     ('run', 'melt', 0.9375),
     ('run', 'streak', 0.9375),
     ('purification', 'refinement', 0.9285714285714286),
     ('internal', 'home', 0),
     ('smother', 'put_out', 0),
     ('doorkeeper', 'usher', 0.9090909090909091),
     ('heel', 'cad', 0.9333333333333333),
     ('made', 'produce', 0.875),
     ('made', 'throw', 0.8888888888888888),
     ('made', 'draw', 0.9),
     ('made', 'score', 0.9166666666666666),
     ('patronize', 'support', 0.9166666666666666),
     ("prairie_bird's-foot_trefoil", 'compass_plant', 0.0),
     ('act', 'turn', 0.9166666666666666),
     ('shimmy', 'shift', 0.8333333333333334),
     ('head', 'lead', 0.9230769230769231),
     ('head', 'guide', 0.9230769230769231),
     ('head', 'point', 0.9375),
     ('connotation', 'intension', 0.0)]




```python
G=nx.DiGraph()
G.add_weighted_edges_from(eds)

st="graph among %i words and their synonyms" %len(G)
g_diagnostics(G,st)
```

    The graph among 125 words and their synonyms has 125 nodes and 195 edges 
    
    The graph among 125 words and their synonyms is a directed graph
    The graph among 125 words and their synonyms graph is a weighted graph
    The graph among 125 words and their synonyms graph is not strongly connected and it has 45 strongly connected components
    The largest strongly connected component of this graph has 39 nodes and 88 edges
    The graph among 125 words and their synonyms graph is not weakly connected and it has 24 weakly connected components
    The largest weakly connected component of this graph has 63 nodes and 125 edges
    The density of the graph among 125 words and their synonyms is 0.013
    The transitivity of the graph among 125 words and their synonyms is 0.107
    The reciprocity of the graph among 125 words and their synonyms is 0.882



```python
edge_width=[G[u][v]['weight'] for u,v in G.edges()]
edge_width=[w if type(w)==float else 0 for w in edge_width]
edge_width=[1*math.log(1.3+w) for w in edge_width]

nsi=[]
for n in G.nodes():
    if G.in_degree(n)>0:
        nsi.append(10*math.log(1+G.in_degree(n)))
    else:
        nsi.append(20)
figsize=(17,13)
pos=graphviz_layout(G) 

labels={}
for n in G.nodes():
    labels[n]=""
    
node_color="#ffb3b3"
node_border_color="r"
edge_color="#668cff"
plt.figure(figsize=figsize);
nodes = nx.draw_networkx_nodes(G, pos, node_color=node_color,node_size=nsi)
nodes.set_edgecolor(node_border_color)
nx.draw_networkx_edges(G, pos,arrowsize=12, width=edge_width,edge_color=edge_color,alpha=0.6)
# nx.draw_networkx_labels(G,pos,labels=labels)
# nx.draw_networkx_edge_labels(G,pos,edge_labels=elabels);
plt.axis('off');
yoffset = {}
y_off = -10 # offset on the y axis
for k, v in pos.items():
    yoffset[k] = (v[0], v[1]+y_off)
nx.draw_networkx_labels(G, yoffset,font_size=13);
# st1="graph of %i words \n in a random sample of %i words and their synonyms" %(len(G),nos)
sst="The directed %s" %st
plt.title(sst,fontsize=20);
plt.margins(x=0.1, y=0) 
```


![png](/images/synonym_nets_files/synonym_nets_12_0.png)



```python
from IPython.display import Image
Image(filename='allDirected.png',width=800, height=400)
```




![png](/images/synonym_nets_files/synonym_nets_13_0.png)




```python
giant = max(nx.weakly_connected_components(G), key=len)
Glwcc=G.subgraph(giant)

st1="weakly connected component of the directed %s" %st
# graph of %i words \n in a random sample of %i words and their synonyms" %(len(G),nos)
g_diagnostics(Glwcc,st1)
```

    The weakly connected component of the directed graph among 125 words and their synonyms has 63 nodes and 125 edges 
    
    The weakly connected component of the directed graph among 125 words and their synonyms is a directed graph
    The weakly connected component of the directed graph among 125 words and their synonyms graph is a weighted graph
    The weakly connected component of the directed graph among 125 words and their synonyms graph is not strongly connected and it has 13 strongly connected components
    The largest strongly connected component of this graph has 39 nodes and 88 edges
    The weakly connected component of the directed graph among 125 words and their synonyms is a weakly connected graph
    The density of the weakly connected component of the directed graph among 125 words and their synonyms is 0.032
    The transitivity of the weakly connected component of the directed graph among 125 words and their synonyms is 0.110
    The reciprocity of the weakly connected component of the directed graph among 125 words and their synonyms is 0.896



```python
edge_width=[Glwcc[u][v]['weight'] for u,v in Glwcc.edges()]
edge_width=[w if type(w)==float else 0 for w in edge_width]
edge_width=[7*math.log(1.3+w) for w in edge_width]

nsi=[]
for n in Glwcc.nodes():
    if Glwcc.in_degree(n)>0:
        nsi.append(20*math.log(1+Glwcc.in_degree(n)))
    else:
        nsi.append(20)
figsize=(17,13)
pos=graphviz_layout(Glwcc) 

labels={}
for n in Glwcc.nodes():
    labels[n]=""
    
node_color="#ffb3b3"
node_border_color="r"
edge_color="#668cff"
plt.figure(figsize=figsize);
nodes = nx.draw_networkx_nodes(Glwcc, pos, node_color=node_color,node_size=nsi)
nodes.set_edgecolor(node_border_color)
nx.draw_networkx_edges(Glwcc, pos,arrowsize=30,width=edge_width, edge_color=edge_color,alpha=0.6)
# nx.draw_networkx_labels(Glscc,pos,labels=labels)
# nx.draw_networkx_edge_labels(G,pos,edge_labels=elabels);
plt.axis('off');
yoffset = {}
y_off = -4 # offset on the y axis
for k, v in pos.items():
    yoffset[k] = (v[0], v[1]+y_off)
nx.draw_networkx_labels(Glwcc, yoffset,font_size=13);
st1="The weakly connected component of the directed %s" %st
# graph of %i words \n in a random sample of %i words and their synonyms" %(len(G),nos)
sst=st1
plt.title(sst,fontsize=20);
plt.margins(x=0.1, y=0) 
```


![png](/images/synonym_nets_files/synonym_nets_15_0.png)



```python
Image(filename='allDirectedLSCC.png',width=800, height=400)
```




![png](/images/synonym_nets_files/synonym_nets_16_0.png)




```python
reds=[]
for e in G.edges(data=True):
    if (e[1],e[0]) in G.edges():
        reds.append(e)
print(len(reds)) #,len(set(reds)))
# for e in reds:
#     if (e[1],e[0]) in reds:
#         reds.remove(e)
# print(len(reds)) #,len(set(reds)))
```

    172



```python
Gr=nx.Graph()
Gr.add_weighted_edges_from(reds)

st1="subgraph of reciprocating references among %i words and their synonyms" %len(Gr)
g_diagnostics(Gr,st1)
```

    The subgraph of reciprocating references among 105 words and their synonyms has 105 nodes and 86 edges 
    
    The subgraph of reciprocating references among 105 words and their synonyms is an undirected graph
    The subgraph of reciprocating references among 105 words and their synonyms graph is a weighted graph
    The subgraph of reciprocating references among 105 words and their synonyms graph is a disconnected graph and it has 25 connected components
    The largest connected component of this graph has 39 nodes and 44 edges
    The density of the subgraph of reciprocating references among 105 words and their synonyms is 0.016
    The transitivity of the subgraph of reciprocating references among 105 words and their synonyms is 0.115



```python
edge_width=[Gr[u][v]['weight']['weight'] for u,v in Gr.edges()]
edge_width=[w if type(w)==float else 0 for w in edge_width]
edge_width=[7*math.log(1.3+w) for w in edge_width]

nsi=[]
for n in Gr.nodes():
    if Gr.degree(n)>0:
        nsi.append(100*math.log(1+Gr.degree(n)))
    else:
        nsi.append(20)
figsize=(17,13)
pos=graphviz_layout(Gr) 

labels={}
for n in Gr.nodes():
    labels[n]=""
    
node_color="lime"
node_border_color="orange"
edge_color="darkblue"
plt.figure(figsize=figsize);
nodes = nx.draw_networkx_nodes(Gr, pos, node_color=node_color,node_size=nsi)
nodes.set_edgecolor(node_border_color)
nx.draw_networkx_edges(Gr, pos,width=edge_width,edge_color=edge_color,alpha=0.6)
# nx.draw_networkx_labels(G, pos)
# nx.draw_networkx_edge_labels(G,pos,edge_labels=elabels);
plt.axis('off');
yoffset = {}
y_off = -15 # offset on the y axis
for k, v in pos.items():
    yoffset[k] = (v[0], v[1]+y_off)
nx.draw_networkx_labels(Gr, yoffset,font_size=13);
# st1="subgraph of %i reciprocating references to synonyms \n in the %s" %(len(Gr),st)
sst="The undirected %s" %st1
plt.title(sst,fontsize=20);
plt.margins(x=0.1, y=0.1) 
```


![png](/images/synonym_nets_files/synonym_nets_19_0.png)



```python
Image(filename='allReci.png',width=800, height=400)
```




![png](/images/synonym_nets_files/synonym_nets_20_0.png)




```python
giant = max(nx.connected_components(Gr), key=len)
Grlscc=Gr.subgraph(giant)

st2="largest connected component of the undirected %s" %st1
# graph of reciprocating synonyms of %i words \n in a random sample of %i words and their synonyms" %(len(G),nos)
g_diagnostics(Grlscc,st2)
```

    The largest connected component of the undirected subgraph of reciprocating references among 105 words and their synonyms has 39 nodes and 44 edges 
    
    The largest connected component of the undirected subgraph of reciprocating references among 105 words and their synonyms is an undirected graph
    The largest connected component of the undirected subgraph of reciprocating references among 105 words and their synonyms graph is a weighted graph
    The largest connected component of the undirected subgraph of reciprocating references among 105 words and their synonyms is a connected graph
    The density of the largest connected component of the undirected subgraph of reciprocating references among 105 words and their synonyms is 0.059
    The transitivity of the largest connected component of the undirected subgraph of reciprocating references among 105 words and their synonyms is 0.140



```python
edge_width=[Grlscc[u][v]['weight']['weight'] for u,v in Grlscc.edges()]
edge_width=[w if type(w)==float else 0 for w in edge_width]
edge_width=[7*math.log(1.3+w) for w in edge_width]

nsi=[]
for n in Grlscc.nodes():
    if Grlscc.degree(n)>0:
        nsi.append(100*math.log(1+Grlscc.degree(n)))
    else:
        nsi.append(20)
figsize=(17,13)
pos=graphviz_layout(Grlscc) 

node_color="lime"
node_border_color="orange"
edge_color="darkblue"
plt.figure(figsize=figsize);
nodes = nx.draw_networkx_nodes(Grlscc, pos, node_color=node_color,node_size=nsi)
nodes.set_edgecolor(node_border_color)
nx.draw_networkx_edges(Grlscc, pos,width=edge_width,edge_color=edge_color,alpha=0.6)
# nx.draw_networkx_labels(G, pos)
# nx.draw_networkx_edge_labels(G,pos,edge_labels=elabels);
plt.axis('off');
yoffset = {}
y_off = -4 # offset on the y axis
for k, v in pos.items():
    yoffset[k] = (v[0], v[1]+y_off)
nx.draw_networkx_labels(Grlscc, yoffset,font_size=13);
# st3="The largest connected component of the undirected %s" %st1
st3="The largest connected component of the \n undirected subgraph of reciprocating references among %i words" %len(Gr)
# subgraph of \n %i reciprocating references to synonyms \n in the %s" %(len(Grlscc),st)
sst=st3
plt.title(sst,fontsize=20);
plt.margins(x=0.1, y=0.1) 
```


![png](/images/synonym_nets_files/synonym_nets_22_0.png)



```python

```
