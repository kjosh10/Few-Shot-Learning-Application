

from Sub_Window_Ml_Model import *

###############################     Creating Class to always create new Window   ###############################

class Download_Resize_Window(Main_Home_Window):
    
    
    def __init__(self, website_name, window, height, width, no_classes, images_per_class):
        self.window = window
        self.HEIGHT = height
        self.WIDTH = width
        self.no_classes = no_classes
        self.website_name = website_name
        self.images_per_class = images_per_class
        self.ask_question_for_classes()
        
    ###############################     Creating Canvas, Frames, Labels, Entries and Buttons   ###############################
    
    def ask_question_for_classes(self):    
        
        canvas = tk.Canvas(self.window, height=self.HEIGHT, width=self.WIDTH)
        canvas.pack()
#         global classes, entries, entries_is_downloaded
        self.classes = []
        self.entries, self.entries_is_downloaded = [], []
        frame_1 = tk.Frame(self.window, bg='#80c1ff', bd=5)
        frame_1.place(relx=0.01, rely=0.01, relwidth=0.98, relheight=0.6)

        for i in range(self.no_classes):
            entry = tk.Entry(frame_1, font=60)
    #         print(0.01+i*0.9/no_classes, i)
            entry.place(relx=(0.01+i*0.98/self.no_classes), rely=0.05, relwidth=(0.98/self.no_classes), relheight=0.2)
            self.entries.append(entry)

        def save_data(entries, entries_is_downloaded):
            for entry, entry_is_saved in zip(entries, entries_is_downloaded):
                self.classes.append(entry.get())
                entry_is_saved.delete(0, END)
                entry_is_saved.insert(0, 'Saved')
    #             entry.delete(0, END)

        button_1 = Button(frame_1, text="Save classes", font=40, command=lambda: save_data(self.entries, self.entries_is_downloaded))
        button_1.place(relx=0.01, rely=0.5, relwidth=0.98, relheight=0.2)

        def download_images(entries_is_downloaded, website_name, classes, images_per_class, token_num):
            if self.website_name == 'Google':
                self.images_downloader_from_google(self.entries_is_downloaded, self.classes, self.images_per_class)
            else:
                self.images_downloader_from_unsplash(self.entries_is_downloaded, self.classes, self.images_per_class, token_num)

        button_2 = Button(frame_1, text="Download Images", font=40, command=lambda: download_images(
            self.entries_is_downloaded, self.website_name, self.classes, self.images_per_class, token_num=1))
        button_2.place(relx=0.01, rely=0.75, relwidth=0.98, relheight=0.2)

        #################################  Resize Options  #################################

        frame_2 = tk.Frame(self.window, bg='#80c1ff', bd=5)
        frame_2.place(relx=0.01, rely=0.635, relwidth=0.98, relheight=0.34)

        label = tk.Label(frame_2, text="Enter Resize shape:-", font=40)
        label.place(relx=0.01, rely=0.08, relwidth=0.46, relheight=0.36)

        entry_resize_shape = tk.Entry(frame_2, font=60)
        entry_resize_shape.place(relx=0.52, rely=0.08, relwidth=0.47, relheight=0.36)

        def resize_images(classes, total_images, images_per_class, shape):
#             global images, labels, resize_shape
            self.resize_shape = int(shape)
            self.images, self.labels = self.image_resizer(classes, total_images, images_per_class, self.resize_shape)

            #################   Normalizing the Images  ###############################
            self.images = self.images/255
            self.labels = self.labels.astype(int)

# #             global X_train, X_test, y_train, y_test
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.images, self.labels, test_size=0.15, random_state=42)
            window_ML_Model = Tk()
            window_ML_Model.attributes('-fullscreen', True)
            window_ML_Model.title('Train ML model')
            
            # Bind the ESC key with the callback function
            window_ML_Model.bind('<Escape>', lambda e: Main_Home_Window.close_win(self, e, window_ML_Model))
            
            self.window.destroy()
            parameters = [self.website_name, window_ML_Model, self.HEIGHT, self.WIDTH, self.no_classes, 
                self.images_per_class, self.classes, self.labels, self.resize_shape, 
                         self.X_train, self.X_test, self.y_train, self.y_test]
            ML_Window_In_App = ML_Model_Window(parameters)
            
            window_ML_Model.mainloop()
#             image_viewer_and_ML_model_trainer(window_ML_Model, HEIGHT, WIDTH, no_classes)


    #     global entry_resized
    #     entry_resized = tk.Entry(frame_2, font=40)
    #     entry_resized.place(relx=0.53, rely=0.48, relwidth=0.46, relheight=0.44)

        button_3 = Button(frame_2, text="Resize Images", font=40, command=lambda: resize_images(
            self.classes, self.no_classes*self.images_per_class, self.images_per_class, entry_resize_shape.get()))
        button_3.place(relx=0.01, rely=0.48, relwidth=0.98, relheight=0.44)

        for i in range(self.no_classes):
            entry_is_downloaded = tk.Entry(frame_1, font=60)
    #         print(0.01+i*0.9/no_classes, i)
            entry_is_downloaded.place(relx=(0.01+i*0.98/self.no_classes), rely=0.25, relwidth=(0.98/self.no_classes), relheight=0.2)
            self.entries_is_downloaded.append(entry_is_downloaded)
    
    ###############################     Methods to Download Images from Google       ###############################
    
    def images_downloader_from_google(self, entries_is_downloaded, classes, images_per_class, token_num=1):
        # Second Section: Declare important variables
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
    
        ### List of Links  ###
        links = []
        
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
    #     print(total_images)
    #     print('<', end =' ')
    #     print('Please wait while downloading images might take some while...')

        for idx, key in enumerate(classes):

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
    #                     print(count)
                        if(count > images_per_class):
                            break

                    except KeyError:
                        continue

    #             print(links[0])

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = [executor.submit(download_from_url, url, key) for url in links]

                end = time.perf_counter()
    #             print('Downloaded {} images of class {} in {} second(s)'.format(len(links), key, round(end - start, 2)))
                entries_is_downloaded[idx].delete(0,END)
    #             print(idx)
                entries_is_downloaded[idx].insert(0, f"Downloaded in {round(end - start, 2)} second(s)")
    #             entries_is_downloaded[idx].set(f'Downloaded')


            except OSError:
                entries_is_downloaded[idx].delete(0, END)
                entries_is_downloaded[idx].insert(0, 'Error!!!')
    #             print('Check your Internet connection.')

    #             links.append(1)

    # #     print(' >', end =' ')
        ### Back to original working directory ###
        os.chdir("..")

    ###############################     Methods to Download Images from Unsplash       ###############################
    
    def images_downloader_from_unsplash(self, entries_is_downloaded, classes, images_per_class, token_num):

        access_token = {0:'RyH2ALP-5KeNjh5y7WKakVe2rvLJqCu9DTPfnTixRWc', 1: 'aJccbYqOTjjP1EXKqf0nlCs9HTTVrdBLyOsmchu7BYw', 2:'THeZYVZEtbMp6LrfVVe4bJmaV2dH5PTwym7oqW22ngo', 3:'VMxWawzk5bBIUMS1NnlPa_XgOwmWfFZ9ekat3Ato88I'}

        ### To save Images in a different Folder ###
        folder_name = 'Downloaded_Images'

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        else:
            shutil.rmtree(os.getcwd()+'\\Downloaded_Images')
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
                                entries_is_downloaded[idx].delete(0, END)
                                entries_is_downloaded[idx].insert(0, 'Error!!!')
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
    #             print(idx)
                entries_is_downloaded[index].delete(0, END)
                entries_is_downloaded[index].insert(0, f"Downloaded in {round(end - start, 2)} second(s)")
    #             print('Downloaded {} images of class {} in {} second(s)'.format(images_per_class, key, round(end - start, 2)))
    #     print(' >', end =' ')

        ### Back to original working directory ###
        os.chdir("..")

    ###############################     Resize Images and save them into Numpy arrays and label them       ###############################
    
    def image_resizer(self, classes, total_images, images_per_class, resize_shape):
        from PIL import Image
        resize_shape = int(resize_shape)
        ### To save Images in a different Folder ###
        folder_name = 'Resized_Images'

        path = f"{os.getcwd()}\\Downloaded_Images"

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        else:
            shutil.rmtree(os.getcwd()+'\\Resized_Images')
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
                img = Image.open(path + f"\\{img_name}")
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



