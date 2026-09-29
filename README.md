# Fake-News-Detector
pip install streamlit scikit-learn pandas



import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ওয়েব অ্যাপের শিরোনাম
st.title("📰 ফেক নিউজ ডিটেক্টর")
st.write("যেকোনো খবরের টেক্সট বা শিরোনাম দিয়ে পরীক্ষা করে দেখুন এটি আসল নাকি ভুয়া।")

# টেস্ট ডেটাসেট (মডেল ট্রেনিংয়ের জন্য সহজ কিছু নমুনা)
texts = [
    "Government announces new education policy for schools",
    "Scientists discover new planet near earth",
    "Aliens landed in New York city yesterday night",
    "Drink this juice to cure all diseases overnight",
    "Health ministry releases new guidelines for wellness"
]
labels = ["Real", "Real", "Fake", "Fake", "Real"]

# মডেল ট্রেনিং (Text Vectorization + Naive Bayes Classifier)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

# ইউজার ইনপুট নেওয়ার ঘর
user_input = st.text_area("খবরের টেক্সট বা হেডলাইন এখানে লিখুন:", "")

# বাটন এবং প্রেডিকশন
if st.button("যাচাই করুন"):
    if user_input.strip() == "":
        st.warning("অনুগ্রহ করে কিছু টেক্সট লিখুন।")
    else:
        # ইনপুট প্রসেস ও প্রেডিকশন
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)[0]
        
        # ফলাফল প্রদর্শন
        if prediction == "Real":
            st.success("✅ এটি একটি **সত্যি খবর (Real News)** মনে হচ্ছে।")
        else:
            st.error("🚨 এটি একটি **ভুয়া খবর (Fake News)** হতে পারে!")
