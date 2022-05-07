
from Sub_Window_Prediction_WebCam import *

###############################     Creating Class to always create new Window   ###############################

class ML_Model_Window(Main_Home_Window):
    
    
    
    def __init__(self, parameters):# website_name, window, height, width, no_classes, images_per_class, classes, labels, resize_shape):
        
        from PIL import ImageTk, Image
        
        self.website_name = parameters[0]
        self.window = parameters[1]
        self.HEIGHT = parameters[2]
        self.WIDTH = parameters[3]
        self.no_classes = parameters[4]
        
        self.images_per_class = parameters[5]
        self.classes = parameters[6]
        self.labels = parameters[7]
        self.resize_shape = parameters[8]
        self.X_train, self.X_test, self.y_train, self.y_test = parameters[9], parameters[10], parameters[11], parameters[12]
        self.flag_top_level=True
#         super().view_images_and_labels(self.window, self.image_list, self.flag_top_level, self.classes, self.labels)
        self.image_viewer_and_ML_model_trainer(self.window, self.HEIGHT, self.WIDTH, self.no_classes)

    
    ###############################     Making the Window Layout                 ###############################
    
    def image_viewer_and_ML_model_trainer(self, window_ML_Model, HEIGHT, WIDTH, no_classes):
        from PIL import ImageTk, Image
        canvas = tk.Canvas(window_ML_Model, height=HEIGHT, width=WIDTH)
        canvas.pack()

        frame_1 = tk.Frame(window_ML_Model, bg='#80c1ff', bd=5)
        frame_1.place(relx=0.01, rely=0.01, relwidth=0.98, relheight=0.98)
        
        



        button_1 = Button(frame_1, text="View Images with their labels", font=40, command=
                          lambda: Main_Home_Window.view_images_and_labels(self, self.window, self.X_train, 
                                                                          self.flag_top_level, self.classes, self.y_train, 
                                                                          self.no_classes, self.images_per_class))
        button_1.place(relx=0.01, rely=0.01, relwidth=0.98, relheight=0.25)
        
        ###############################     Method to Train the model              ###############################
        
        def train_model(no_epochs, entry_epoch):
#             global model_VGG16
            self.model_VGG16 = models.Sequential()
            self.epochs = no_epochs
        
            ## VGG16, ResNet50, MobileNetV2
            pretrained_model = tf.keras.applications.VGG16(weights = 'imagenet', include_top = False, 
                                                           input_shape = (self.resize_shape,self.resize_shape,3), pooling=None)  # Optional 152

            for layer in pretrained_model.layers:
                layer.trainable = False

            self.model_VGG16.add(pretrained_model)
            self.model_VGG16.add(layers.Flatten())
            # model_VGG16.add(layers.Dense(512, activation='relu'))
            self.model_VGG16.add(layers.Dense(self.no_classes, activation='softmax'))


            self.model_VGG16.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

            #global history
            self.history = self.model_VGG16.fit(self.X_train, self.y_train,epochs=int(no_epochs), 
                                                validation_data=(self.X_test, self.y_test), verbose=0)

            entry_epoch.delete(0, END)
            entry_epoch.insert(0, "Model trained")


        label = tk.Label(frame_1, text="Number of Epochs", font=40)
        label.place(relx=0.01, rely=0.3, relwidth=0.25, relheight=0.25)

        entry_epoch = tk.Entry(frame_1, font=40)
        entry_epoch.place(relx=0.3, rely=0.3, relwidth=0.35, relheight=0.25)

        button_2 = Button(frame_1, text="Train VGG16 Model", font=40, command=lambda: train_model(entry_epoch.get(), entry_epoch))
        button_2.place(relx=0.7, rely=0.3, relwidth=0.29, relheight=0.25)

        ###############################     Method to plot accuracy plot                ###############################
        
        def plot_accuracy(history):
            window_plot_accuracy = Tk()
            # setting the title 
            window_plot_accuracy.title('Plotting in Tkinter')

            # dimensions of the main window
            window_plot_accuracy.geometry("500x500")

            # the figure that will contain the plot
            fig = Figure(figsize = (5, 5), dpi = 100)

            # adding the subplot
            plot_accuracy = fig.add_subplot(111)

            # plotting the graph
            plot_accuracy.plot(history.history['accuracy'])
            plot_accuracy.plot(history.history['val_accuracy'])
            plot_accuracy.set_title('Training Accuracy vs Validation Accuracy')
            plot_accuracy.set_xlabel('Epochs')
            plot_accuracy.set_ylabel('Accuracy')
            plot_accuracy.legend(['Train', 'Val'])

            # creating the Tkinter canvas
            # containing the Matplotlib figure
            canvas = FigureCanvasTkAgg(fig,
                                       master = window_plot_accuracy)  
            canvas.draw()

            # placing the canvas on the Tkinter window
            canvas.get_tk_widget().pack()

            # creating the Matplotlib toolbar
            toolbar = NavigationToolbar2Tk(canvas,
                                           window_plot_accuracy)
            toolbar.update()

            # placing the toolbar on the Tkinter window
            canvas.get_tk_widget().pack()

            window_plot_accuracy.mainloop()

        ###############################     Method to plot losses                 ###############################
        
        def plot_loss(history):
            window_plot_loss = Tk()
            # setting the title 
            window_plot_loss.title('Plotting in Tkinter')

            # dimensions of the main window
            window_plot_loss.geometry("500x500")

            # the figure that will contain the plot
            fig = Figure(figsize = (5, 5), dpi = 100)

            # adding the subplot
            plot_loss = fig.add_subplot(111)

            # plotting the graph
            plot_loss.plot(history.history['loss'])
            plot_loss.plot(history.history['val_loss'])
            plot_loss.set_title('Training Loss vs Validation Loss')
            plot_loss.set_xlabel('Epochs')
            plot_loss.set_ylabel('Loss')
            plot_loss.legend(['Train','Val'])

            # creating the Tkinter canvas
            # containing the Matplotlib figure
            canvas = FigureCanvasTkAgg(fig,
                                       master = window_plot_loss)  
            canvas.draw()
            
            # placing the canvas on the Tkinter window
            canvas.get_tk_widget().pack()

            # creating the Matplotlib toolbar
            toolbar = NavigationToolbar2Tk(canvas,
                                           window_plot_loss)
            toolbar.update()

            # placing the toolbar on the Tkinter window
            canvas.get_tk_widget().pack()

            window_plot_loss.mainloop()

        button_3 = Button(frame_1, text="Show Accuracy plot", font=40, command=lambda: plot_accuracy(self.history))
        button_3.place(relx=0.01, rely=0.58, relwidth=0.47, relheight=0.25)

        button_4 = Button(frame_1, text="Show Loss plot", font=40, command=lambda: plot_loss(self.history))
        button_4.place(relx=0.52, rely=0.58, relwidth=0.47, relheight=0.25)

        ###############################     Command method to go to the next Window                 ###############################
        
        def go_next():
            window_prediction_camera = Tk()
            window_prediction_camera.attributes('-fullscreen', True)
            window_prediction_camera.title('WebCam Prediction')
    #         window_prediction_camera = tk._default_root
            self.window.destroy()
            self.path = os.getcwd()
            
            # Bind the ESC key with the callback function
            window_prediction_camera.bind('<Escape>', lambda e: Main_Home_Window.close_win(self, e, window_prediction_camera))
            
            parameters = [window_prediction_camera, self.HEIGHT, self.WIDTH, self.no_classes, 
                self.images_per_class, self.classes, self.labels, self.resize_shape, 
                         self.X_train, self.X_test, self.y_train, self.y_test, self.model_VGG16, self.path, self.epochs]
            WebCam_Predictor_Window = Prediction_With_WebCam(parameters)
            
            window_prediction_camera.mainloop()
#             predictor_camera(model_VGG16, window_prediction_camera, HEIGHT, WIDTH, no_classes)

            
        button_5 = Button(frame_1, text="Next", font=40, command=lambda: go_next())
        button_5.place(relx=0.01, rely=0.85, relwidth=0.98, relheight=0.14)

#         window_ML_Model.mainloop()

    #     button_back = Button()
    
        
        