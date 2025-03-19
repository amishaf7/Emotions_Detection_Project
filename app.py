import streamlit as st 
import joblib
import neattext.functions as nfx 

#Load the saved model and vectorizer
model = joblib.load("best_emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

#Emotion Mapping (Adjust labels as per data)
emotion_mapping = {
    0: "Sad 😢",
    1: "Happy 😊",
    2: "Angry 😡",
    3: "Surprised 😲",
    4: "Fearful 😨",
    5: "Neutral 😐"
}

#Emotion Prediction Function
def predict_emotion(text):
  text_cleaned = nfx.remove_special_characters(text)
  text_cleaned = nfx.remove_stopwords(text_cleaned)
  text_vectorized = vectorizer.transform([text_cleaned])
  prediction = model.predict(text_vectorized)[0]
  return emotion_mapping.get(prediction,"Unknow Emotion🤔 ")

#Streamlit UI
st.set_page_config(page_title="Emotion Detector",page_icon="🎭")
st.title("Emotion Detection App 🎭")
st.subheader("Analyze the emotion behind your text using AI !")
st.write("Simply enter a sentence, and the model will predict the emotion behind it.")

user_input = st.text_area("Enter your text here:","")

#Button
if st.button("Predict Emotion"):
  if user_input:
    emotion = predict_emotion(user_input)
    st.success(f"The Predicted Emotion is: **{emotion}**")
  else:
    st.warning("Please enter some text!")

#Footer
st.markdown("-----")
st.markdown("**Made with ❤️ by Amisha Mishra**")
