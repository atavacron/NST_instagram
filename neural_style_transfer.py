from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import date, datetime
import json
import random
from instabot import Bot 
import google_api
import csv

tf.random.set_seed(2020)

content_layers = [['block5_conv2', "block3_conv3"], 
                  ['block4_conv2'], 
                  ['block4_conv2', "block5_conv2"], 
                  ['block5_conv2'], 
                  ["block5_conv3"]]

styles_layers = [["block1_conv1",
                "block2_conv1", 
                "block3_conv1", 
                "block4_conv1", 
                "block5_conv1"],
                ["block1_conv2",
                "block2_conv2", 
                "block3_conv2", 
                "block4_conv2", 
                "block5_conv2"],
                ["block1_conv1",
                "block2_conv1", 
                "block1_conv2", 
                "block2_conv2", 
                "block3_conv2"],
                ["block1_conv1",
                "block2_conv1", 
                "block1_conv2", 
                "block2_conv2", 
                "block3_conv3",
                "block4_conv3"]]

def image_read(image_path):
    """
    Process image, respahe the image with 3 channels
    """
    #this read the bytes of the image
    img = tf.io.read_file(image_path)
    #channels stands for colors RGB
    img = tf.image.decode_image(img, channels=3)
    #converts image to float
    img = tf.image.convert_image_dtype(img, tf.float32)
    #cast to a new type
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    #value to scale when pass shape
    scale = 720 / long_dim
    #integer shape times scale
    new_shape = tf.cast(shape * scale, tf.int32)
    #resize image with new shape
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return(img)

def vgg_layers(layer_names):
    """
    Creates the vgg model using layers_names layers.
    """
    #VGG19 architecture https://arxiv.org/pdf/1409.1556.pdf. Image net for pretrained
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    #trainable is always going to be false
    vgg.trainable = False
    #architecture of the model vgg using layer names, should be a list
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    """
    Define the gram matrix function. Core of Neural style transfer.
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return(result/(num_locations))

def style_content_loss(outputs, style_targets, style_weight, dims_style, content_targets, content_weight, dims_content):
    """
    Compute the loss
    """
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) for name in style_outputs.keys()])
    style_loss *= style_weight / dims_style

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
    content_loss *= content_weight / dims_content
    loss = style_loss + content_loss
    return(loss)

def tensor_to_image(tensor):
    """
    Returned tensor to image
    """
    #255 is the default input to VGG
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        #if tensor.shape[0] == 1 then pass else assertion error
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def clip_0_1(image):
    """
    clip values to min and max
    """
    return(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

def reading_json(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
    return(data)

def random_selection(hyperparameter):
    #for learning rate
    if hyperparameter == "lr":
        big_number = random.uniform(0, 1)
        #with 30% probabilitpick learning rate between 0.011 and 0.2 
        #with 70% probability pick from 0 to 0.01
        if big_number > 0.7:
            final = random.uniform(0.011, 0.2)
        else:
            final = random.uniform(0.001, 0.01)
    elif hyperparameter == "beta_1":
        final = random.uniform(0.5, 0.9)
    elif hyperparameter == "negative":
        if random.uniform(0, 1) > 0.6:
            final = float("1e" + str(random.randint(0, 3)))
        else:
            if random.uniform(0, 1) > 0.25:
                final = float("10e-"+ str(random.randint(1, 3)))
            else:
                final = float("10e-"+ str(random.randint(4, 6)))
    elif hyperparameter == "positive":
        if random.uniform(0, 1) > 0.6:
            final = float("1e-" + str(random.randint(0, 3)))
        else:
            if random.uniform(0, 1) <= 0.95:
                final = float("1e" + str(random.randint(0, 3)))
            else:
                final = float("1e" + str(random.randint(3, 6)))
    elif hyperparameter == "epochs":
        big_number = random.uniform(0, 1)
        if big_number < 0.3:
            final = random.randint(3, 5)
        else:
            final = random.randint(6, 10)
    return(final)

def final_caption(caption, lr, beta1, epochs, style_weight, content_weight, content_layer, style_layers):
    lr = str(lr)
    epochs = str(epochs)
    beta1 = str(beta1)
    style_weight = str(style_weight)
    content_weight = str(content_weight)
    content_layer = str(content_layer)
    style_layers = str(style_layers)
    final_capt = "{} Usando Learning rate de {}, razón de caída exponencial de {} y {} épocas. Peso alpha de contenido {} y peso Beta de estilo {}. Con capas de contenido {} y capas de estilo {}.".format(caption, lr, beta1, epochs, content_weight, style_weight, content_layer, style_layers)
    return(final_capt)

def append_csv(to_append, path_to_write):
    with open(path_to_write, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(to_append)
    print("Data append succesfully")

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layer):
        super(StyleContentModel, self).__init__()
        #build the model using vgg_layers function
        self.vgg = vgg_layers(style_layers + content_layer)
        #style and content layer
        self.style_layers = style_layers
        self.content_layer = content_layer
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
    
    def call(self, inputs):
        #since standard input of vgg is 255
        inputs = inputs * 255.0
        #adequate the image to the model
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        #extract style and content 
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        #compute the gram matrix for the style output
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        #create content and style dict
        content_dict = {content_name:value for content_name, value in zip(self.content_layer, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content':content_dict, 'style':style_dict}

def main():
    upload = True
    #reset all state generated by keras
    tf.keras.backend.clear_session()
    #call two download both images to prod file
    style_pic, content_pic = google_api.main()
    #from json file specyfing caption depends on style pic 
    caption = reading_json("style_name.json")[style_pic]
    print("Images have been downloaded")
    style_img = image_read("prod_folder/style.jpg")
    content_img = image_read("prod_folder/content.jpg")
    #will use one block for content (V1 has fixed NN layers)
    content_layer = random.choice(content_layers)
    style_layers = random.choice(styles_layers)
    
    #dimensions for content and style
    dims_content = len(content_layer)
    dims_style = len(style_layers)
    #specify layers to define vgg
    extractor = StyleContentModel(style_layers, content_layer)
    #extract content from first layers
    results = extractor(tf.constant(content_img))
    #extract style
    style_targets = extractor(style_img)["style"]
    #extract content
    content_targets = extractor(content_img)["content"]
    #variable for inmutability
    image = tf.Variable(content_img)
    #select Adam optimizer
    lr = random_selection("lr")
    beta1 = random_selection("beta_1")
    optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=beta1, epsilon=1e-1)
    #beta
    style_weight = random_selection("negative") 
    #alpha
    content_weight = random_selection("positive") 

    #keep track of time
    start = time.time()
    epochs = random_selection("epochs")
    steps_per_epoch = 100
    print(final_caption(caption, lr, beta1, epochs, style_weight, content_weight, content_layer, style_layers))

    @tf.function()
    def train_step(image):
        #automatic differentiation
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            #calculate loss
            loss = style_content_loss(outputs, style_targets, style_weight, dims_style, content_targets, content_weight, dims_content)
        grad = tape.gradient(loss, image)
        optimizer.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    #keep track of steps
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            #10*100
            step += 1
            train_step(image)
            print(".", end='')
            if m % 25 == 0:
                try:
                    tf.debugging.check_numerics(image, "all_null", name=None)
                except:
                    upload = False
                    break 
        #tensor_to_image(image).save("output.png")
        if upload == False:
            break
        else:
            image_from_tensor = tensor_to_image(image).resize((320,320), Image.ANTIALIAS)
            output_temp_path = 'output/output_log/'+str(start) + ".jpg"
            image_from_tensor.save(output_temp_path)
        print("Train step: {}".format(step)) 
    
    end = time.time()
    total_time = end-start

    #transform and save the image
    if upload:
        image_from_tensor = tensor_to_image(image).resize((1080,1080), Image.ANTIALIAS)
        output_path = 'output/'+str(date.today()) + ".jpg"
        image_from_tensor.save(output_path) 
    
    #perform instagram manipulation 
    username = "username"
    password = "password"
    if upload:
        pass
        bot = Bot()
        bot.login(username = username, password = password)
        bot.upload_photo(output_path, caption = final_caption(caption, lr, beta1, epochs, style_weight, content_weight, content_layer, style_layers))
    else:
        pass

    store_data = [style_pic, 
                  content_pic, 
                  str(date.today()), 
                  lr, 
                  beta1, 
                  epochs,
                  style_weight, 
                  content_weight, 
                  str(date.today()) + ".jpg", 
                  total_time, 
                  str(content_layer), 
                  str(style_layers),
                  str(upload)]

    append_csv(store_data, "logs.csv")

    os.remove("prod_folder/content.jpg")
    os.remove("prod_folder/style.jpg")
    print("Files Removed!")
    if upload == False:
        raise TypeError("nan or inf in tensor")
    else:
        upload = True
    return(upload)

if __name__ == '__main__':
    val = False
    while val == False:
        try:
            val = main()
        except:
            val = False