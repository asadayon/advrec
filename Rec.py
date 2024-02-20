import streamlit as st
import pandas as pd
import subprocess
import json




if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

st.title("Advisor Recommender System ")
name=st.text_input('Enter your Name')
keywords=st.text_area('Enter keywords of your reseach interest')

st.button('Submit', on_click=click_button)

if st.session_state.clicked:
    with st.chat_message("Assistant"):
        st.write("Hello "+name+"! Please wait while we retrieve some information.")
        import time
        with st.spinner(text='In progress'):
            process = subprocess.Popen(['python', 'TFIDF_recommender.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output, error = process.communicate(input=keywords)

            # Check for errors
            if process.returncode == 0:
                #print(output)
                data_dict={}
                data_dict = json.loads(output)
                
                df = pd.DataFrame(data_dict)
                st.dataframe(df)
                with open('rec_result.txt', 'w') as f:
                    msg="User name is "+ name+"User reseach interests are "+keywords+". Top 3 recommended advisor list:\n"
                    for i in range(len(data_dict['Ranking'])):
                        msg+=str(i)+'. name: '+ data_dict['Name'][i]
                        msg+='. Similarity score'+str(data_dict['Similarity Score'][i])
                        msg+='. Keywords: '+data_dict['Keywords'][i]+'\n'
                    f.write(msg)
                
                #st.write(f"Result from receiver.py: {output}")
            else:
                st.write(f"Error in receiver.py: {error}")
           
            #df = pd.DataFrame(data)
            #st.write(output.strip())
            st.success('Done')
#def main_page():
#    st.markdown("# Main page 🎈")
#    st.sidebar.markdown("# Main page 🎈")

#def page2():
#    st.markdown("# Page 2 ❄️")
#    st.sidebar.markdown("# Page 2 ❄️")

#def page3():
#    st.markdown("# Page 3 🎉")
#    st.sidebar.markdown("# Page 3 🎉")


#selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
#Advisor_Rec()
