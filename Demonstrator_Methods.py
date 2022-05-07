

    ############################################     Necessary Imports     ############################################

import tkinter as tk
from tkinter import *

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk# , NavigationToolbar2TkAgg
import cv2
import os
import requests
import urllib.request as url
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from bs4 import BeautifulSoup
import shutil
import time
import concurrent.futures
import random
from PIL import ImageTk, Image

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model

    
    ###############################     Methods to Download Images from Google       ###############################
    
def images_downloader_from_google(classes, images_per_class):
    
    google_image = "https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&"

    user_agent = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"
    }

    def download_from_url(link, key):
        response = requests.get(link)
        image_name = key + '-' + str(random.randint(1, 1e11)) + ".jpg"
#         image_name = key + '-' + link.split(':')[2].split('-')[0].split('=')[0] + '.jpg'

        with open(image_name, 'wb') as fh:
            fh.write(response.content)


    ### To save Images in a different Folder ###
    folder_name = 'Downloaded_Images'
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    else:
        shutil.rmtree(os.getcwd()+'\\Downloaded_Images')
        os.mkdir(folder_name)

    os.chdir(os.getcwd()+f'//{folder_name}')

    total_images = images_per_class * len(classes)
    total_img_downloaded = 0
    multiplier = 0

    for idx, key in enumerate(classes):
        
        ### List of Links  ###
        links = []
        
        
        img_downloaded_per_class = 0
        page_num = 1

        search_url = google_image + 'q=' + key
#         print(search_url)
        try:
            response = requests.get(search_url, headers=user_agent)
             #     print(response)
            html = response.text
        #     print(html)
            soup = BeautifulSoup(html, 'html.parser')
        #     print(soup)
            results = soup.findAll('img', {'class': 'rg_i Q4LuWd'})

            count = 1

            start = time.perf_counter()

            for result in results:
                try:
                    link = result['data-src']
                    links.append(link)
                    count += 1
                    if(count > images_per_class):
                        break

                except KeyError:
                    continue

#             print(links[0])

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = [executor.submit(download_from_url, url, key) for url in links]

            end = time.perf_counter()


        except OSError:
            print('Check your Internet connection.')

    ### Back to original working directory ###
    os.chdir("..")
        
        
        
    ###############################     Methods to Download Images from Unsplash       ###############################

def images_downloader_from_unsplash(classes, images_per_class, token_num):

    access_token = {0:'RyH2ALP-5KeNjh5y7WKakVe2rvLJqCu9DTPfnTixRWc', 1: 'aJccbYqOTjjP1EXKqf0nlCs9HTTVrdBLyOsmchu7BYw', 2:'THeZYVZEtbMp6LrfVVe4bJmaV2dH5PTwym7oqW22ngo', 3:'VMxWawzk5bBIUMS1NnlPa_XgOwmWfFZ9ekat3Ato88I'}

    ### To save Images in a different Folder ###
    folder_name = 'Downloaded_Images'

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    else:
        shutil.rmtree(os.getcwd()+'/Downloaded_Images')
        os.mkdir(folder_name)

    os.chdir(os.getcwd()+f'//{folder_name}')

#     print('Please wait while downloading images might take some while...')

    total_images = images_per_class * len(classes)
    total_img_downloaded = 0
    multiplier = 0

    def download_from_url(img_url, key):

        file_name = key + '-' + str(random.randint(1, 1e11)) + ".jpg"
#         file_name = key + '-' + img_url.split('-')[1] + ".jpg"
#         print(file_name, key)
        url.urlretrieve(img_url, file_name)

#     print(total_images)
#     print('<') # , end =' '
    for index, key in enumerate(classes):

        img_downloaded_per_class = 0
        page_num = 1
        start = time.perf_counter()
        while (img_downloaded_per_class <= images_per_class):

            try:
                if token_num == 5:
                    break
                request = requests.get(f'https://api.unsplash.com/search/photos?page={page_num}&query={key}&per_page=30&client_id={access_token[token_num-1]}')
            except OSError as e:
#                 print('Error Handled')
                token_num += 1
                try:
                    request = requests.get(f'https://api.unsplash.com/search/photos?page={page_num}&query={key}&per_page=30&client_id={access_token[token_num-1]}')
                except OSError as e:
#                     print('Error Handled')
                    token_num += 1
                    try:
                        request = requests.get(f'https://api.unsplash.com/search/photos?page={page_num}&query={key}&per_page=30&client_id={access_token[token_num-1]}')
                    except OSError as e:
#                         print('Error Handled')
                        token_num += 1
                        try:
                            request = requests.get(f'https://api.unsplash.com/search/photos?page={page_num}&query={key}&per_page=30&client_id={access_token[token_num-1]}')
                        except OSError as e:
#                             print('Error Handled')
                            token_num += 1
#                             print('Check your Internet Connection.')
                            break
#             print(token_num)

            data = request.json()
            urls = []
            for idx, img_data in enumerate(data['results']):
                if img_downloaded_per_class >= images_per_class:
                    break
                img_downloaded_per_class += 1
                total_img_downloaded += 1
                file_name = key + '_' + str(img_downloaded_per_class) + ".jpg"
                img_url = img_data['urls']['raw']

                img_url = img_data['urls']['raw']

#                 print(f'Image {img_downloaded_per_class} saved of class {key}...')
                suffix = '&q=80&fm=jpg&crop=entropy&cs=tinysrgb&w=720&fit=max'
#                 print(img_url)
                img_url = img_url + suffix

                urls.append(img_url)
#               url.urlretrieve(img_url, file_name)


#                     if int(total_img_downloaded * 100/ total_images) % 5 == 0:
#                         sys.stdout.write('.')
#                         sys.stdout.flush()
#             print(len(urls), key)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = [executor.submit(download_from_url, url, key) for url in urls]
#                 executor.map(download_from_url, urls)

            if img_downloaded_per_class >= images_per_class:
                break    
#             print(urls)

            page_num += 1

        if token_num != 5:
            end = time.perf_counter()   
            
#     print(' >', end =' ')

    ### Back to original working directory ###
    os.chdir("..")
    
    
    
    ###############################     Resize Images and save them into Numpy arrays and label them       ###############################

def image_resizer(classes, total_images, images_per_class, resize_shape):
    from PIL import Image
    resize_shape = int(resize_shape)
    ### To save Images in a different Folder ###
    folder_name = 'Resized_Images'

    path = f"{os.getcwd()}/Downloaded_Images"

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    else:
        shutil.rmtree(os.getcwd()+'/Resized_Images')
        os.mkdir(folder_name)

    os.chdir(os.getcwd()+f'//{folder_name}')

#     print('Please wait while all the images are being resized...')
    total_images = len(classes) * images_per_class
    images = np.zeros((total_images, resize_shape, resize_shape, 3))
    labels = np.zeros(total_images)
    image_counter = 0

    for idx, img_name in enumerate(os.listdir(path)):
        ## If image ##
#         print(img_name)
        if img_name.split('.')[1] == 'jpg': 
#             print(img_name, idx-1)
            image_counter += 1
#             print(image_counter)
            if (image_counter > images_per_class):
                image_counter = 1
            file_name = img_name.split('-')[0] + '_' + str(image_counter) + '.jpg'
#             print(file_name, idx, img_name)
            img = Image.open(path + f"/{img_name}")
            img = img.resize((resize_shape, resize_shape))
#             img.save(file_name)
#             print(np.asarray(img).shape)

            if len(np.array(img).shape) == 2:
            ###########  If Gray scale image, then flip horizontally the precious image  ########
                images[idx] = np.fliplr(images[idx-1])
                labels[idx] = labels[idx-1]
#                 print(images[idx].shape)
#                 plt.imshow(images[idx]/255)
#                 img = Image.fromarray(images[idx])
                cv2.imwrite(file_name, images[idx])
            else:
                images[idx] = np.asarray(img)
                labels[idx] = (np.where(np.array(classes) == img_name.split('-')[0]))[0][0]
                img.save(file_name)
    #         print(img_name)
#     print('Resizing completed.')
#     entry_resized.delete(0, END)
#     entry_resized.insert(0, "Resized")
    os.chdir("..")

    return images, labels



    ###############################     Method to Train the model              ###############################

def train_modelVGG16(classes, X_train, X_test, y_train, y_test, epochs, resize_shape, verbose=0):
    
    model_VGG16 = models.Sequential()

    ## VGG16
    pretrained_model = tf.keras.applications.VGG16(weights = 'imagenet', include_top = False, 
                                                   input_shape = (resize_shape, resize_shape,3), pooling=None)  # Optional 152

    for layer in pretrained_model.layers:
        layer.trainable = False

    model_VGG16.add(pretrained_model)
    model_VGG16.add(layers.Flatten())
    # model_VGG16.add(layers.Dense(512, activation='relu'))
    model_VGG16.add(layers.Dense(len(classes), activation='softmax'))


    model_VGG16.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    #global history
    history = model_VGG16.fit(X_train, y_train,epochs=int(epochs), validation_data=(X_test, y_test), verbose=verbose)

    return model_VGG16



    #############################     Command method to turn on the Web Cam and start real time predictions ###############################
    
    
def image_snapper_video(resize_shape, model, classes):

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
    img = img.resize((resize_shape, resize_shape))
#         img.save(img_name)
    img_np = np.asarray(img)
#     plt.imshow(img_np)
    img_np = img_np.reshape((1, resize_shape, resize_shape, 3))
    y_pred = model.predict(img_np)
    text = classes[np.argmax(y_pred)]

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
            img = img.resize((resize_shape, resize_shape))

            img_np = np.asarray(img)

            img_np = img_np.reshape((1, resize_shape, resize_shape, 3))
            y_pred = model.predict(img_np)

            text = classes[np.argmax(y_pred)]

        for idx, class_name in enumerate(classes):
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
    
def database_manager(classes, images_per_class, epochs, model, feedback, X_test, y_test):
    
    import sqlite3
    
    classes_string = ""
    
    #    Command to add the details to the data base
    
    def submit():

        #   Create a Database or connect to one   
        conn = sqlite3.connect('Model_history.db')

        #   Create a cursor
        c = conn.cursor()

        #    Insert into the table
        c.execute("INSERT INTO addresses VALUES (:classes_string, :imgs, :no_epochs, :accuracy, :user_feedback)",
            {
                'classes_string':classes_string,
                'imgs':images_per_class,
                'no_epochs': epochs,
                'accuracy':model.evaluate(X_test, y_test, verbose=0)[1],
                'user_feedback': feedback
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
    print(f"Classes {classes_string} were used and {images_per_class} images per class was downloaded. \nModel was trained on {epochs} epochs and the test accuracy was : {model.evaluate(X_test, y_test, verbose=0)[1]}")

    
    
    
    for idx, class_name in enumerate(classes):
        if idx < (len(classes) - 2):
            classes_string = classes_string + class_name +", "
        elif idx < (len(classes) - 1):
            classes_string = classes_string + class_name +" "
        else:
            classes_string = classes_string + 'and ' + class_name

    if not os.path.isfile('Model_history.db'):
#         print(os.getcwd())
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

    submit()
    

    