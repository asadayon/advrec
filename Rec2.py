import streamlit as st
import pandas as pd
import json
from nltk.stem import PorterStemmer
import numpy as np
from  scipy.spatial.distance import cosine
from heapq import nsmallest,nlargest
from openai import OpenAI


data = pd.read_csv('my_dataframe.csv')

count_vector={}
with open('my_dict.json', 'r') as f:
        count_vector = json.load(f)

Term_set=[]

with open('Term_set.json', 'r') as f:
        Term_set = json.load(f)

def tokenize(txt):
  txt=str(txt)
  txt = txt.replace(';',' ')
  return txt.split()

def porter_stemmer(words):
  stemmer = PorterStemmer()
  return [stemmer.stem(word) for word in words]

def user_count_vector(doc): 
    lst=[]
    doc=tokenize(doc)
    doc=porter_stemmer(doc)   
    for j in Term_set:
         lst.append(doc.count(j))
    return lst
def top_similar_doc_cosine(count_vec,doc,k=3):
        lst={}
        for i in count_vec:
            lst[i]=1-cosine(count_vec[i],user_count_vector(doc))
        top_similar_doc = nlargest(k, lst, key = lst.get)
        return lst,top_similar_doc


def cosine_recommender(doc):
    # Read data from stdin
    

    user_score, rec_adv=top_similar_doc_cosine(count_vector,doc,3)
    user_score, rec_adv=top_similar_doc_cosine(count_vector,doc,3)
    rank=[]
    user=[]
    score=[]
    kw=[]
    count=1
    for i in rec_adv:
                   rank.append(count)
                   user.append(i)
                   score.append(user_score[i])
                   a=data[data['n']==i]['t']
                   for j in a: 
                        k=" ".join(tokenize(j))
                        #k=tokenize(j)
                   kw.append(k)
                   count+=1
    df = {
                                        'Ranking': rank,
                                        'Name': user,
                                        'Similarity Score': score,
                                        'Keywords': kw
                                    }
    data_str = json.dumps(df)
    
    #print("Done")
    with open('rec_result.json', 'w') as f:
        json.dump(df, f)
    return data_str

def load_dict(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def social_network(dic):
    rec=[]
    df={}
    r_name=[]
    r_t=[]
    lst=dic['Name']
    print(lst)
    kw=dic['Keywords']
    data=pd.read_csv("dataframe_id.csv")
    soc_id=load_dict('top3_id.json')
    soc_val=load_dict('top3_val.json')    
    for i in range(len(lst)):
        name=lst[i]
        key=kw[i]
        for index,row in data.iterrows():
            name2=row['n']
            idx=row['id']
            a=row['t']
            for j in a: 
                        k=" ".join(tokenize(j))

            if name2==name:
                if key==k:
                    print("Found ID: ",idx, " of ", name)
                    break
        if str(float(idx)) in soc_id:
            soc_i=soc_id[str(float(idx))]
            soc_v=soc_val[str(float(idx))]
            rec.append(soc_i[soc_v.index(min(soc_v))])
    for idx in rec:
        for index,row in data.iterrows():
            name2=data['n']
            idx2=data['id']
            a=data['t']
            for j in a: 
                        k=" ".join(tokenize(j))
            if idx==idx2:
                r_name.append(name2)
                r_t.append(k)
    df = {
                                        
                                        'Name': r_name,
                                        
                                        'Keywords': r_t
                                    }
    data_str = json.dumps(df)
    
    #print("Done")
    with open('soc_result.json', 'w') as f:
        json.dump(df, f)
    return data_str
        
        
            
            
            
            
        



def LDA(keywords):

    from gensim import corpora, models,  similarities
    import ast
    import string

    data = pd.read_csv('my_dataframe.csv')
    lda_model = models.LdaModel.load('lda_model.model')
    dictionary = corpora.Dictionary.load('dictionary.dict')
    index = similarities.MatrixSimilarity.load('index_file.index')
    rank=[]
    top=[]
    topic_words=[]
    topic_prob=[]
    names=[]
    sim=[]
    kw=[]


    #new_doc = ['end', 'user', 'parallel', 'program', 'liter', 'program', 'key', 'practic', 'program', 'style', 'end-us', 'softwar', 'engin', 'softwar', 'develop', 'end-us', 'program', 'situat', 'program', 'experi', 'program', 'standard']
    new_doc=keywords
    punc=''',;.'''
    for i in punc:
        if i in new_doc:
            new_doc.replace(i,'')
    new_doc=new_doc.split()
    new_doc=porter_stemmer(new_doc)  
    
    new_doc_bow = dictionary.doc2bow(new_doc)

    new_doc_distribution = lda_model.get_document_topics(new_doc_bow)
    sorted_doc_topics = sorted(new_doc_distribution, key=lambda x: -x[1])
    top3_topics = sorted_doc_topics[:1]
    for topic, prob in top3_topics:
        print(f"Topic {topic} with probability {prob}")
        top_words = lda_model.show_topic(topic, 10)
        words_only = [word for word, pro in top_words]
        print(f"Top words for topic {topic}: {', '.join(words_only)}")
        top.append(topic)
        topic_words.append(words_only)
        topic_prob.append(prob)
        
        
        
        
    query_lda = lda_model[new_doc_bow]
    sims = index[query_lda]    
    sims_list = list(enumerate(sims))
    sorted_sims = sorted(sims_list, key=lambda item: -item[1])
    top3_documents = sorted_sims[:3]
    count=1
    for doc_position, score in top3_documents:
        print(f"Document id: {doc_position}, name: {data['n'][doc_position]} with similarity score: {score}")
        rank.append(count)
        names.append(data['n'][doc_position])
        a=data['t'][doc_position]
        #a=data[data['n']==i]['t']
        a=a.replace(";"," ")
                  
        kw.append(a)
        print(a)
        sim.append(score)
        count+=1
    df1 = {
                                        'LDA_rank': rank,
                                        'LDA_Name': names,
                                        'Score': sim,
                                        'Keywords_LDA': kw
                                       
                                    }
    #data_str1 = json.dumps(df1)
    
    #print("Done")
    #with open('LDA_rec_result.json', 'w') as f:
     #   json.dump(df1, f)
        
        
    df2 = {
                                        'Topic': top,
                                        'Words': topic_words,
                                        'Probability': topic_prob
                                       
                                    }
    #data_str2 = json.dumps(df2)
    
    #print("Done")
    #with open('LDA_topic_result.json', 'w') as f:
     #   json.dump(df2, f)
    return df1,df2
        
        



if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state.clicked = True


st.title("Advisor Recommender System ")
name=st.text_input('Enter your Name')
keywords=st.text_area('Enter keywords of your reseach interest')

st.button('Submit', on_click=click_button)
data_dict={}
flag=0
if st.session_state.clicked:
    #with st.chat_message("Assistant"):
        st.session_state.clicked = False
        st.write("Hello "+name+"! Please wait while we retrieve some information.")
        import time
        with st.spinner(text='In progress'):
            output=cosine_recommender(keywords)           
            data_dict = json.loads(output)
            
            lda1,lda2=LDA(keywords)
            soc=social_network(data_dict)
            soc = json.loads(soc)
            #lda1=json.loads(lda1)
            #lda2=json.loads(lda2)
            
            if "openai_model" not in st.session_state:
                st.session_state["openai_model"] = "gpt-3.5-turbo"
            if "flag" not in st.session_state:          
                st.session_state["flag"] = data_dict
                with open('rec_result.txt', 'w') as f:
                        msg="User name is "+ name+". User reseach interests are "+keywords+". Top 3 recommended advisor list based on Cosine similarity:\n"
                        for i in range(len(data_dict['Ranking'])):
                            msg+=str(i+1)+'. name: '+ data_dict['Name'][i]
                            msg+='. Cosine similarity score: '+str(data_dict['Similarity Score'][i])
                            msg+='. Keywords: '+data_dict['Keywords'][i]+'\n'
                        f.write(msg)
            if "lda1" not in st.session_state:
                st.session_state["lda1"] = lda1
                with open('rec_result.txt', 'a') as f:
                        msg="\nTop 3 recommended advisor list based on LDA Topic modeling:\n"
                        for i in range(len(lda1['LDA_rank'])):
                            msg+=str(i+1)+'. name: '+ lda1['LDA_Name'][i]
                            #msg+='. Cosine similarity score: '+str(data_dict['Similarity Score'][i])
                            msg+='. Keywords: '+lda1['Keywords_LDA'][i]+'\n'
                        f.write(msg)
            if "lda2" not in st.session_state:
                st.session_state["lda2"] = lda2
                with open('rec_result.txt', 'a') as f:
                        msg="\nTop LDA Topic selected:\n"
                        for i in range(len(lda2['Topic'])):
                            msg+=str(i+1)+'. Topic id: '+ str(lda2['Topic'][i])
                            #msg+='. Cosine similarity score: '+str(data_dict['Similarity Score'][i])
                            msg+='. Keywords: '+" ".join(lda2['Words'][i])+'\n'
                        f.write(msg)
            if "soc" not in st.session_state:
                st.session_state["soc"] = soc
                with open('rec_result.txt', 'a') as f:
                        msg="\nTop recommended advisor list based on Social modeling:\n"
                        for i in range(len(soc['Name'])):
                            msg+=str(i+1)+'. name: '+ soc['Name'][i]
                            msg+='. Keywords: '+soc['Keywords'][i]+'\n'
                        f.write(msg)

            if "messages" not in st.session_state:
                msg="" 
                with open('rec_result.txt', 'r') as f:
                                for line in f:
                                    msg+=line
                              
                st.session_state.messages = [{'role':'system', 'content':msg+"""
            You are an Advisor Recommendation System, you will explain why any advisor is recommended to the user. Recommendations were given based on the cosine similarity and LDA Topic similarity of user research interest and advisor research interest keywords. \
            You are given a username and three recommended advisor names to the user. You are also provided with user research interests, and research interests of three recommended advisors. \
            You may explain based on research interests are similar, and why. If the user asks about the cosine similarity score, which has a value of 0 to 1, you may use different examples to describe the similarity score.\
            You respond in a short, very conversational friendly style. 
            """}]
                response="Welcome "+name+" to the Advisor Recommender System. Would you like an explaination of your recommendation for advisors?"
                st.session_state.messages.append({"role": "assistant", "content": response})
if "flag" in st.session_state:
    df1 = pd.DataFrame(st.session_state["flag"])
    df2 = pd.DataFrame(st.session_state["lda1"])
    df3 = pd.DataFrame(st.session_state["lda2"])
    df4 = pd.DataFrame(st.session_state["soc"])
    
    df1_new = df1[['Ranking', 'Name']]
    st.write("Top 3 recommended advisor based on Cosine Similarity of keywords:")
    st.dataframe(df1_new)
    st.write("Top 3 recommended advisor based on LDA Topic Similarity of 20 topics:")
    df2_new = df2[['LDA_rank','LDA_Name' ]]
    st.dataframe(df2_new)
    
    if len(df4)!=0:          
        df4_new = df3[['Name']]
        st.write("Top recommended advisor based on Social connectivity:")
        st.dataframe(df4_new)
    else:
        st.write("Didn't find any from Social connectivity:")

    
            


client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
if "messages"  in st.session_state:
    for message in st.session_state.messages:
        if message['role']=='system':
            continue
        if message['role']=='user':
            with st.chat_message(message["role"],avatar="👦"):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


    if prompt := st.chat_input("Type here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user",avatar="👦"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
                

               



