import base64
from io import BytesIO
import cv2
import streamlit as st

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default .xml') #a pre-trained model that can be used to detect faces in images and videos

def detect_faces():
    
    # Initialize the web cam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        st.warning('Cannot open camera. Check your permission settings!!!', icon="⚠️")
        cap.release()
        exit()
    
    while True:
        # Read the frame from the webcam
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            st.warning('Cannot receive frame (stream end?). Exiting ...', icon="⚠️")
            break
        
        # Convert the image to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image using the pre-trained model
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the frame
        st.image(frame, caption='Face Detection using Viola-Jones Algorithm', channels='RGB')
    
        if 113 == int(ord('q')):
            return frame
        
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
    
# def get_image_download_link(img,filename,text):
#     buffered = BytesIO()
#     img.save(buffered, format="JPEG")
#     img_str = base64.b64encode(buffered.getvalue()).decode()
#     href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
#     return href
    
def save_image(img):
    cv2.imwrite('images/face_detection.png', img)
    
    
def app():
    html_title_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Detection using Viola-Jones Algorithm </h2>
    </div>
    """
    st.markdown(html_title_temp, unsafe_allow_html = True)
    
    st.write('Press the button below to start detecting faces from your webcam')
    
    if st.button('Detect Faces'):
        image = detect_faces()
        
        if st.button("Save Image", on_click=save_image(image)):
            print('gets here?')
            st.success('Image saved successfully!!!')
        
        
if __name__ == '__main__':
    app()