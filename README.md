# Fake-News-Detector
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ওয়েব অ্যাপের শিরোনাম
st.title("Fake News Detector")
st.write("যেকোনো খবরের টেক্সট বা শিরোনাম দিয়ে পরীক্ষা করে দেখুন এটি আসল নাকি ভুয়া।")

texts = [
    # --- Real News (সত্যি খবর) ---
    "NASA launches new satellite to study climate change",
    "Local library opens new section for young readers",
    "Scientists find new species of deep sea fish",
    "Central bank updates interest rates for savings accounts",
    "City council approves funding for new public park",
    "WHO publishes report on global health improvements",
    "High school students win national science competition",
    "Rainfall brings relief to drought affected region",
    "Tech company introduces energy efficient laptop",
    "New highway bridge opens to reduce traffic congestion",
    "University researchers develop faster water filter",
    "Local farmers market expands operating hours",
    "Astronomers observe distant galaxy with powerful telescope",
    "Public transport system introduces electric buses",
    "Museum hosts special exhibition on ancient artifacts",
    "Weather department predicts heavy snowfall in mountains",
    "Medical trial shows promising results for heart treatment",
    "Firefighters safely rescue family from burning building",
    "Government signs agreement to build renewable solar plants",
    "Sports authority organizes youth marathon event",
    "New study links balanced diet with better focus",
    "City launches recycling campaign for plastic waste",
    "Hospitals receive new medical equipment for emergency rooms",
    "Wildlife sanctuary reports increase in rare bird population",
    "Engineers construct storm resistant coastal barrier",

    # --- Fake News (ভুয়া খবর) ---
    "Eating garlic every hour makes you permanently immune to viruses",
    "Secret underground city discovered beneath Atlantic ocean",
    "Drinking seawater will double your brain intelligence",
    "Scientists confirm moon is made entirely of ancient cheese",
    "Using smartphones for five minutes causes instant blindness",
    "Flying cars now available in local supermarkets for ten dollars",
    "Robots secretly replace all teachers in high schools today",
    "Placing onions in your shoes cures all fever instantly",
    "Time traveler from year 3000 visits local coffee shop",
    "Giant sea monster spotted resting near city harbour",
    "Sleeping next to plants absorbs all your energy overnight",
    "Mysterious crystal found in backyard grants free electricity forever",
    "Aliens broadcast daily news channel on television",
    "Walking backwards for one hour burns ten thousand calories",
    "Lions found living secretly in rainforest canopy",
    "Chewing plastic gum turns your teeth into solid gold",
    "Bathing in vinegar guarantees total protection from lightning",
    "New smartphone app allows you to talk directly with pets",
    "Ancient wooden map reveals location of infinite treasure",
    "Rain falling on Tuesdays contains pure liquid silver",
    "Eating chocolate before sleep makes you float in air",
    "Supermarket orange juice turns completely into milk at midnight",
    "Drinking boiled grass eliminates the need for sleep forever",
    "Clouds are made of cotton candy according to new discovery",
    "Wearing red shoes makes you run faster than a sports car"
]

labels = [
    # Real labels (২৫ টি)
    "Real", "Real", "Real", "Real", "Real",
    "Real", "Real", "Real", "Real", "Real",
    "Real", "Real", "Real", "Real", "Real",
    "Real", "Real", "Real", "Real", "Real",
    "Real", "Real", "Real", "Real", "Real",
    
    # Fake labels (২৫ টি)
    "Fake", "Fake", "Fake", "Fake", "Fake",
    "Fake", "Fake", "Fake", "Fake", "Fake",
    "Fake", "Fake", "Fake", "Fake", "Fake",
    "Fake", "Fake", "Fake", "Fake", "Fake",
    "Fake", "Fake", "Fake", "Fake", "Fake"
]
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
