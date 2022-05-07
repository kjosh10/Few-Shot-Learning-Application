

class Prediction_With_WebCam(Main_Home_Window):
    
    def __init__(self, parameters):# website_name, window, height, width, no_classes, images_per_class, classes, labels, resize_shape):
        
        from PIL import ImageTk, Image
        
        self.window = parameters[0]
        self.HEIGHT = parameters[1]
        self.WIDTH = parameters[2]
        self.no_classes = parameters[3]
        
        self.images_per_class = parameters[4]
        self.classes = parameters[5]
        self.labels = parameters[6]
        self.resize_shape = parameters[7]
        self.X_train, self.X_test, self.y_train, self.y_test = parameters[8], parameters[9], parameters[10], parameters[11]
        self.model = parameters[12]
        self.path = parameters[13]
        self.epochs = parameters[14]
        self.flag_top_level=False
        self.y_test_predict_model = self.model.predict(self.X_test)
        self.predictor_camera()
        
    ###############################     Method to setup the Window Layout                 ###############################
    
    def predictor_camera(self):
        
        canvas = tk.Canvas(self.window, height=self.HEIGHT, width=self.WIDTH)
        canvas.pack()
        from PIL import ImageTk, Image
        

        frame_1 = tk.Frame(self.window, bg='#80c1ff', bd=5)
        frame_1.place(relx=0.01, rely=0.02, relwidth=0.98, relheight=0.32)
        
        self.y_test_predict = [np.argmax(self.y_test[idx]) for idx, y_test in enumerate(self.y_test_predict_model)]
        
        button_view_predictions = Button(frame_1, text="View Model predictions", font=40, command=
                                         lambda: Main_Home_Window.view_images_and_labels(self, self.window, self.X_test, 
                                                                          self.flag_top_level, self.classes, self.y_test_predict, 
                                                                          self.no_classes, self.images_per_class))
        button_view_predictions.place(relx=0.01, rely=0.2, relwidth=0.28, relheight=0.6)

        def open_camera_predictor():
            self.image_snapper_video()


        button_open_camera = Button(frame_1, text="Open Camera for Real time predictions", font=40, command=lambda: 
                                    open_camera_predictor())
        button_open_camera.place(relx=0.34, rely=0.2, relwidth=0.28, relheight=0.6)

        button_exit = Button(frame_1, text="Exit APP", font=40, command=lambda: self.window.destroy())
        button_exit.place(relx=0.67, rely=0.2, relwidth=0.31, relheight=0.6)
        
        frame_2 = tk.Frame(self.window, bg='#80c1ff', bd=5)
        frame_2.place(relx=0.01, rely=0.38, relwidth=0.98, relheight=0.60)
        
        self.database_manager(frame_2)
        
    #############################     Command method to turn on the Web Cam and start real time predictions ###############################
    
    
    def image_snapper_video(self):

        def returnCameraIndexes():
            # checks the first 10 indexes.
            index = 0
            arr = []
            i = 10
            while i > 0:
                if index == 0:
                    cap = cv2.VideoCapture(index)
                else:
                    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

                if cap.read()[0]:
                    arr.append(index)
                    cap.release()
                index += 1
                i -= 1
            return arr

        arr = returnCameraIndexes()

        from PIL import ImageTk, Image

        if len(arr) == 1:
            cam_port = 0
            cam = cv2.VideoCapture(cam_port)
        else:
            cam_port = returnCameraIndexes()[1]
            cam   = cv2.VideoCapture()
            cam.open(cam_port, cv2.CAP_DSHOW)

        count = 0
        img_name = 'Img_from_Cam.png'
        ret, frame = cam.read()
        
        # saving image in local storage
        cv2.imwrite(img_name, frame)

        img = Image.open(os.getcwd() + f"/{img_name}")
        img = img.resize((self.resize_shape, self.resize_shape))
    #         img.save(img_name)
        img_np = np.asarray(img)
    #     plt.imshow(img_np)
        img_np = img_np.reshape((1, self.resize_shape, self.resize_shape, 3))
        y_pred = self.model.predict(img_np)
        text = self.classes[np.argmax(y_pred)]

        coordinates = (10,100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (0,0,255)
        thickness = 2

        while (True):

            ret, frame = cam.read()

            # saving image in local storage
            cv2.imwrite(img_name, frame)

            count += 1

            # Initialize blank mask image of same dimensions for drawing the shapes
            shapes = np.zeros_like(frame, np.uint8)

            if count % 10 == 0:
                img = Image.open(os.getcwd() + f"/{img_name}")
                img = img.resize((self.resize_shape, self.resize_shape))
                
                img_np = np.asarray(img)
                
                img_np = img_np.reshape((1, self.resize_shape, self.resize_shape, 3))
                y_pred = self.model.predict(img_np)
                
                text = self.classes[np.argmax(y_pred)]

            for idx, class_name in enumerate(self.classes):
                x = 10
                y = 300 + idx*30
                coordinates_classes=(x, y)
                text_class_name = class_name + "({:.1f})".format(y_pred[0][idx])
                if class_name == text:
                    cv2.putText(frame, text_class_name, coordinates_classes, font, 0.8, color, thickness, cv2.LINE_AA)
                else:
                    cv2.putText(frame, text_class_name, coordinates_classes, font, 0.8, (0, 128, 255), thickness, cv2.LINE_AA)
                    
                cv2.rectangle(shapes, (180, 283+idx*29), (int(180+y_pred[0][idx]*400), 305+idx*29), (255, 255, 255), cv2.FILLED)

        # Generate output by blending image with shapes image, using the shapes
        # images also as mask to limit the blending to those parts

            cv2.putText(frame, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
            alpha = 0.5
            mask = shapes.astype(bool)
            frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

            cv2.imshow("Frame", frame)
        #         cv2.imshow("Text", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

    ###############################     Create a Database                ###############################
    
    def database_manager(self, frame):
        import sqlite3
        
        self.classes_string = ""
        for idx, class_name in enumerate(self.classes):
            if idx < (len(self.classes) - 2):
                self.classes_string = self.classes_string + class_name +", "
            elif idx < (len(self.classes) - 1):
                self.classes_string = self.classes_string + class_name +" "
            else:
                self.classes_string = self.classes_string + 'and ' + class_name
                
        if not os.path.isfile('Model_history.db'):
            print(os.getcwd())
            #   Create a Database or connect to one   
            conn = sqlite3.connect('Model_history.db')

            #   Create a cursor
            c = conn.cursor()

            #    Create the table
            c.execute("""CREATE TABLE addresses (
                classes text,
                images_per_class integer,
                Epochs integer,
                Model_accuracy real, 
                feedback text
            )
            """)
            
            #    Commit changes
            conn.commit()
            
            #    Close the connection
            conn.close()
        
        
        #    Create Submit button
        submit_button = Button(frame, text="Add to the database and show the summary", font=40, command=lambda: submit(entry_summary))
        submit_button.place(relx=0.01, rely=0.02, relwidth=0.98, relheight=0.3)
        
        #    Create Submit button
        label_feedback = Label(frame, text="Please give your feedback with an adjective :-", font=40)
        label_feedback.place(relx=0.01, rely=0.37, relwidth=0.65, relheight=0.3)
        
        entry_feedback = Entry(frame, font=40)
        entry_feedback.place(relx=0.71, rely=0.37, relwidth=0.28, relheight=0.3)
        
        entry_summary = Entry(frame, font=40)
        entry_summary.place(relx=0.01, rely=0.72, relwidth=0.98, relheight=0.26)
        
        #    Command to add the details to the data base
        
        def submit(entry):
            
            #   Create a Database or connect to one   
            conn = sqlite3.connect('Model_history.db')

            #   Create a cursor
            c = conn.cursor()

            #    Insert into the table
            c.execute("INSERT INTO addresses VALUES (:classes_string, :imgs, :no_epochs, :accuracy, :user_feedback)",
                {
                    'classes_string':self.classes_string,
                    'imgs':self.images_per_class,
                    'no_epochs': self.epochs,
                    'accuracy':self.model.evaluate(self.X_test, self.y_test, verbose=0)[1],
                    'user_feedback': entry.get()
                }
            )
            
            #    Query the Database
            c.execute("SELECT *, oid FROM addresses")
            records = c.fetchall()
#             print(records)
            
            #    Commit changes
            conn.commit()
                
            #    Close the connection
            conn.close()
        
            #    Show Summary
            entry.delete(0, END)
            entry.insert(0, f"Classes {self.classes_string} were used and {self.images_per_class} images per class was downloaded. {self.epochs} epochs was the model trained on and the test accuracy was : {self.model.evaluate(self.X_test, self.y_test, verbose=0)[1]}")
        