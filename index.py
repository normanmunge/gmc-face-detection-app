import cv2
import streamlit as st
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default .xml') #a pre-trained model that can be used to detect faces in images and videos


def hex_to_rbg(hex):
    return tuple(int(hex[i:i+2], 16) for i in (0,2, 4))
    
def detect_faces(hex_color, neighbors = 5, scale = 1.3):
    
    # Initialize the web cam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.warning('Cannot open camera. Check your permission settings!!!', icon="⚠️")
        return
    
    # Prepare the Streamlit placeholder
    frame_placeholder = st.empty()
    captured_frame = None

    
    # Set the color to draw the rectangle around the detected faces
    color = hex_to_rbg(hex_color[1:])
    
    # Loop until the user presses the 'q' key
    while True:
        # Read the frame from the webcam
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret:
            st.warning('Cannot read frame webcam!', icon="⚠️")
            break
        
        # Convert the image to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        # Detect faces in the image using the pre-trained model
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors)
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (color), 2)
        
        # Save the last processed frame for downloading
        captured_frame = frame.copy()
        
        # Display the frame in Streamlit
        frame_placeholder.image(frame, caption='Face Detection', channels='BGR', use_column_width=True)

        # Small delay to reduce CPU usage
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
        
        # Stop if the user presses the "Stop Detection" button
        # if st.button("Stop Detection", key=i):
        #     print('wahalla')
        #     break
        
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
    return captured_frame
    
def save_image(img):
    cv2.imwrite('images/face_detection.png', img)
    
    
def app():
    html_title_temp = """
    <div style="background:#025246 ;padding:10px; margin-bottom:30px">
    <h2 style="color:white;text-align:center;">Face Detection App </h2>
    </div>
    """
    st.markdown(html_title_temp, unsafe_allow_html = True)

    st.sidebar.header("User Parameters")
    hex_color = st.sidebar.color_picker("Pick a rectangle color", "#00FF00")
    neighbors = st.sidebar.slider("Min Neighbors i.e how many neighbors each rectangle should have to retain it.", 1, 10, 5)
    scale = st.sidebar.slider("Scale Factor i.e how much size of the face do you want to be detected?", 1.1, 2.0, 1.3, step=0.1)

    if st.button("Start Detection"):
        detect_faces(hex_color, neighbors, scale)
        
if __name__ == '__main__':
    app()
