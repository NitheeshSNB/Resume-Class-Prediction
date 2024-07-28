import numpy
import nltk
import re
import pickle
import streamlit as st
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfv = TfidfVectorizer(stop_words='english')

nltk.download('punkt')
nltk.download('stopwords')

#loading clf and tfidf
clf = pickle.load(open('clf.pkl','rb'))
tfv = pickle.load(open('tfv.pkl','rb'))

def clean_data(txt):
    clntxt = re.sub('http\S+\s',' ',txt)
    clntxt = re.sub('RT|cc',' ',clntxt)
    clntxt = re.sub('#\S+\s',' ',clntxt)
    clntxt = re.sub("@\S+",' ',clntxt)
    clntxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',clntxt)
    clntxt = re.sub(r'[^\x00-\x7f]',' ',clntxt)
    clntxt = re.sub('\s+',' ',clntxt)
    return clntxt

#webapp
def main():
    st.title("Resume Classification App")
    up_file = st.file_uploader("Upload Your Resume", type=['txt','pdf'])

    if up_file is not None:
        try:
            res_bytes = up_file.read()
            res_txt = res_bytes.decode('utf-8')
        except UnicodeDecodeError:
            res_txt = res_bytes.decode('latin-1')

        cln_res = clean_data(res_txt)
        cln_res = tfv.transform([cln_res])
        pred = clf.predict(cln_res)[0]
        st.write(pred)
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(pred, "Unknown")

        st.write("Resume belongs to", category_name,"Category")
#main
if __name__ == "__main__":
    main()

