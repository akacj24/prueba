import os
#from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import PyPDF2
from PIL import Image as Image, ImageOps as ImagOps
import glob
from gtts import gTTS
import os
import time
from streamlit_lottie import st_lottie
import json
import paho.mqtt.client as mqtt
import pytz


MQTT_BROKER = "157.230.214.127"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor_st"

# Variables de estado para los datos del sensor
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = None

def text_to_speech(text, tld):
                
    tts = gTTS(response,"es", tld , slow=False)
    try:
        my_file_name = text[0:20]
    except:
        my_file_name = "audio"
    tts.save(f"temp/{my_file_name}.mp3")
    return my_file_name, text


                
def remove_files(n):
    mp3_files = glob.glob("temp/*mp3")
    if len(mp3_files) != 0:
      now = time.time()
      n_days = n * 86400
      for f in mp3_files:
         if os.stat(f).st_mtime < now - n_days:
             os.remove(f)




def get_mqtt_message():
    """Función para obtener un único mensaje MQTT"""
    message_received = {"received": False, "payload": None}
    
    def on_message(client, userdata, message):
        try:
            payload = json.loads(message.payload.decode())
            message_received["payload"] = payload
            message_received["received"] = True
        except Exception as e:
            st.error(f"Error al procesar mensaje: {e}")
    
    try:
        client = mqtt.Client()
        client.on_message = on_message
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.subscribe(MQTT_TOPIC)
        client.loop_start()
        
        timeout = time.time() + 5
        while not message_received["received"] and time.time() < timeout:
            time.sleep(0.1)
        
        client.loop_stop()
        client.disconnect()
        
        return message_received["payload"]
    
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None

try:
    os.mkdir("temp")
except:
    pass

with st.sidebar:
    st.subheader("Que es PILL-E?")
    st.write(
    """PILL-E es un asistente médico personal que te ayuda a sanar :)
       
    """
                )            

st.title('Hola!!! Soy PILL-E 💬')
#image = Image.open('Instructor.png')
#st.image(image)




video_path = "pilleanimado.mp4"  # Cambia esto al nombre de tu archivo

# Reproduce el video
st.video(video_path, format="video/mp4")








#ke = st.text_input('Ingresa tu Clave')
#os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['OPENAI_API_KEY'] = st.secrets["settings"]["key"] #ke

#st.write(st.secrets["settings"]["key"])

pdfFileObj = open('Temperaturas.pdf', 'rb')
 
# creating a pdf reader object
pdfReader = PyPDF2.PdfReader(pdfFileObj)


    # upload file
#pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

   # extract the text
#if pdf is not None:
from langchain.text_splitter import CharacterTextSplitter
 #pdf_reader = PdfReader(pdf)
pdf_reader  = PyPDF2.PdfReader(pdfFileObj)
text = ""
for page in pdf_reader.pages:
         text += page.extract_text()

   # split into chunks
text_splitter = CharacterTextSplitter(separator="\n",chunk_size=500,chunk_overlap=20,length_function=len)
chunks = text_splitter.split_text(text)

# create embeddings
embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks, embeddings)

# show user input
#st.subheader("Usa el campo de texto para hacer tu pregunta")
#user_question = st.text_area(" ")
user_question=" "
#if user_question:
#        docs = knowledge_base.similarity_search(user_question)

#        llm = OpenAI(model_name="gpt-4o-mini")
#        chain = load_qa_chain(llm, chain_type="stuff")
#        with get_openai_callback() as cb:
#          response = chain.run(input_documents=docs, question=user_question)
#          print(cb)
#        st.write(response)

        
#    
#        if st.button("Escuchar"):
#          result, output_text = text_to_speech(response, 'es-us')
#          audio_file = open(f"temp/{result}.mp3", "rb")
#          audio_bytes = audio_file.read()
#          st.markdown(f"## Escucha:")
#          st.audio(audio_bytes, format="audio/mp3", start_time=0)



            
#          def remove_files(n):
#                mp3_files = glob.glob("temp/*mp3")
#                if len(mp3_files) != 0:
#                    now = time.time()
#                    n_days = n * 86400
#                    for f in mp3_files:
#                        if os.stat(f).st_mtime < now - n_days:
#                            os.remove(f)
#                            print("Deleted ", f)
            
            
#          remove_files(7)

# Columnas para sensor y pregunta
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Datos del Sensor")
    if st.button("Obtener Lectura"):
        with st.spinner('Obteniendo datos del sensor...'):
            sensor_data = get_mqtt_message()
            st.session_state.sensor_data = sensor_data
            
            if sensor_data:
                st.success("Datos recibidos")
                st.metric("Temperatura", f"{sensor_data.get('Temp', 'N/A')}°C")
                st.metric("Humedad", f"{sensor_data.get('Hum', 'N/A')}%")
            else:
                st.warning("No se recibieron datos del sensor")

#/////////////hola///////////#

st.write("Selecciona tus síntomas de la lista a continuación y haz clic en 'Continuar' para obtener un diagnóstico.")

# Lista de síntomas
sintomas = [
    "Tos", "Congestión nasal", "Dificultad para respirar", "Dolor de garganta",
    "Dolor de cabeza", "Dolor general", "Cansancio", "Diarrea", "Vómito",
    "Escalofríos", "Dolor de estómago", "Picazón en la piel",
    "Picazón en la garganta", "Enrojecimiento de la piel", "Hinchazón"
]

# Agrupar síntomas por categorías de diagnóstico
resfriado_comun = {"Tos", "Congestión nasal", "Dificultad para respirar", "Dolor de garganta", 
                   "Dolor de cabeza", "Dolor general", "Cansancio"}
infeccion_gastrointestinal = {"Diarrea", "Vómito", "Escalofríos", "Dolor de estómago"}
alergia = {"Picazón en la piel", "Picazón en la garganta", "Enrojecimiento de la piel", "Hinchazón"}

# Crear checkbox para cada síntoma
sintomas_seleccionados = []
for sintoma in sintomas:
    if st.checkbox(sintoma):
        sintomas_seleccionados.append(sintoma)

# Evaluar síntomas al presionar el botón "Continuar"
if st.button("Continuar"):
    if not sintomas_seleccionados:
        st.warning("Por favor, selecciona tus síntomas. Pide ayuda a tus padres si es necesario.")
    else:
        # Convertir a conjunto para facilitar la comparación
        sintomas_set = set(sintomas_seleccionados)
        
        # Verificar coincidencia con los diagnósticos
        es_resfriado = sintomas_set.intersection(resfriado_comun)
        es_infeccion_gastro = sintomas_set.intersection(infeccion_gastrointestinal)
        es_alergia = sintomas_set.intersection(alergia)
        
        # Determinar el diagnóstico basado en los síntomas seleccionados
        if (es_resfriado and not es_infeccion_gastro and not es_alergia):
            st.success("Padeces de Resfriado común. Dile a Pill-E pastilla roja.")
            audio_file = open("resfriado.mp3", "rb")
            st.audio(audio_file, format="audio/mp3")
            
            st.subheader("Ir a interfaz de voz")
            st.link_button("Vamos a recoger el medicamento", "https://ctrlvozpill-e-lsczplbqs3zvwegpj4p8ct.streamlit.app/")
        elif (es_infeccion_gastro and not es_resfriado and not es_alergia):
            st.success("Padeces de Infección Gastrointestinal. Dile a Pill-E pastilla morada.")
            audio_file = open("infgastro.mp3", "rb")
            st.audio(audio_file, format="audio/mp3")
            
            st.subheader("Ir a interfaz de voz")
            st.link_button("Vamos a recoger el medicamento", "https://ctrlvozpill-e-lsczplbqs3zvwegpj4p8ct.streamlit.app/")
        elif (es_alergia and not es_resfriado and not es_infeccion_gastro):
            st.success("Padeces de Alergia. Dile a Pill-E pastilla azul.")
            audio_file = open("alergia.mp3", "rb")
            st.audio(audio_file, format="audio/mp3")
            
            st.subheader("Ir a interfaz de voz")
            st.link_button("Vamos a recoger el medicamento", "https://ctrlvozpill-e-lsczplbqs3zvwegpj4p8ct.streamlit.app/")
        else:
            st.warning("No puedo generar un diagnóstico para tu enfermedad, lo mejor sería que fueras al médico.")



        #st.subheader("Pide tu medicamento")
        #st.link_button("Vamos a recoger el medicamento", "https://ctroldevoz-cvu77lovh2bkrzmojjmskv.streamlit.app/")
    
             
#                            print("Deleted ", f)
            
            
#          remove_files(7)


# Cerrar archivo PDF
pdfFileObj.close()

