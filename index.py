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

        # # Add a small delay to avoid freezing
        # time.sleep(0.1)
        
        # time.sleep(1)
        # # Display the frame
        # st.image(frame, caption='Face Detection using Viola-Jones Algorithm', channels='RGB')

        # # Exit the loop when 'q' is pressed

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
        # # Display the frame
        # st.image(frame, caption='Face Detection using Viola-Jones Algorithm', channels='RGB')
    
        # if 113 == int(ord('q')):
        #     print('gets here?')
        #     return frame
        
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
    # html_title_temp = """
    # <div style="background:#025246 ;padding:10px; margin-bottom:30px">
    # <h2 style="color:white;text-align:center;">Face Detection using Viola-Jones Algorithm </h2>
    # </div>
    # """
    # st.markdown(html_title_temp, unsafe_allow_html = True)
    
    # html_subtitle_temp = """
    # <div style="padding:10px; margin-bottom:15px">
    #     <h4 style="font-size: 30px"> Let's get started by selecting parameters to adjust our face detection algorithm </h4>
    # </div>
    # """
    # st.markdown(html_subtitle_temp, unsafe_allow_html = True)
    
    # col1, col2, col3 = st.columns(3)
    
    # # Ask user to select color to draw the rectangle around the detected faces
    # with col1:
    #     st.subheader('Color Picker')
    #     html_color_temp = """
    #     <div style="margin-top:40px"></div>
    #     """
    #     st.markdown(html_color_temp, unsafe_allow_html = True)
    #     hex_color = st.color_picker('Pick a color to draw around the detected face', '#00FF00')
    #     st.write("The current color is", hex_color)
    
    # # minNeighbors
    # with col2:
    #     st.subheader('Neighbor')
    #     minNeigbors = st.slider("Select the minNeighbors i.e how many neighbors each rectangle should have to retain it.?", min_value=0, max_value=8, value=5, step=1)
    #     st.write("The current min neighbor is", minNeigbors)
    
    # # scale
    # with col3:
    #     st.subheader('Scale Picker')
    #     html_scale_temp = """
    #     <div style="margin-top:20px"></div>
    #     """
    #     st.markdown(html_scale_temp, unsafe_allow_html = True)
    #     scaleFactor = st.slider("Select the scale i.e how much size of the face do you want to be detected?", min_value=0.0, max_value=3.0, value=1.05, step=0.05)
    #     st.write("The current scale is", scaleFactor)
    
    # if hex_color and minNeigbors and scaleFactor:
        
    #     html_spacing_temp = """
    #     <div style="margin:0 40px">
    #         <hr class="solid">
    #     </div>
    #     """
    #     st.markdown(html_spacing_temp, unsafe_allow_html = True)
        
    #     st.subheader('Press the button below to start detecting faces from your webcam')
        
    #     # Add a button to start detecting faces
    #     if st.button('Detect Faces'):
    #         image = detect_faces(hex_color, minNeigbors, scaleFactor)
            
    #         if st.button("Save Image", on_click=save_image(image)):
    #             print('gets here to save image?')
    #             st.success('Image saved successfully!!!')
        
        
if __name__ == '__main__':
    app()