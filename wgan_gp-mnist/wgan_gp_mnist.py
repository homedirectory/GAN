import datetime, os, time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, initializers, constraints, optimizers, losses
from tensorflow.keras.preprocessing.image import save_img

#KERNEL_INIT = initializers.RandomNormal(mean=1., stddev=0.02)
KERNEL_INIT = None
IMG_SIDE_LENGTH = 28
NOISE_DIM = 128

class WGAN_GP:
    
    def __init__(self):
        self.models = (self.__build_critic(), self.__build_generator())
        
    def __build_critic(self):
        C = models.Sequential(name="Critic")
        
        # 28 x 28 x 1
        C.add(layers.Conv2D(
            64, (3, 3), strides=(1, 1), input_shape=(IMG_SIDE_LENGTH, IMG_SIDE_LENGTH, 1), padding='same', kernel_initializer=KERNEL_INIT
        ))
        # 28 x 28 x 64
        C.add(layers.LeakyReLU(0.2))
        C.add(layers.Conv2D(
            64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=KERNEL_INIT
        ))
        # 14 x 14 x 64
        C.add(layers.LeakyReLU(0.2))
        C.add(layers.Conv2D(
            128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=KERNEL_INIT
        ))
        # 7 x 7 x 128
        C.add(layers.Flatten())
        # 7*7*128
        C.add(layers.Dense(1))
        # Output linear
        
        return C

    def __build_generator(self):
        G = models.Sequential(name='Generator')

        # in: 128 x 1
        G.add(layers.Dense(7*7*256, input_shape=(NOISE_DIM,), kernel_initializer=KERNEL_INIT))
        # 7*7*256
        G.add(layers.Reshape((7, 7, 256)))
        # 7 x 7 x 256
        G.add(layers.BatchNormalization())
        G.add(layers.ReLU())
        G.add(layers.Conv2DTranspose(
            256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=KERNEL_INIT
        ))
        # 14 x 14 x 256
        G.add(layers.BatchNormalization())
        G.add(layers.ReLU())
        G.add(layers.Conv2DTranspose(
            128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=KERNEL_INIT
        ))
        # 28 x 28 x 128
        G.add(layers.BatchNormalization())
        G.add(layers.ReLU())
        G.add(layers.Conv2DTranspose(
            1, (3, 3), strides=(1, 1), padding='same', kernel_initializer=KERNEL_INIT, activation='tanh'
        ))
        # 28 x 28 x 1
       
        return G 

def generator_loss(gen_scores):
    return -1 * tf.reduce_mean(gen_scores)

def critic_loss(real_scores, gen_scores, grad_penalty):
    return tf.reduce_mean(gen_scores) - tf.reduce_mean(real_scores) + grad_penalty

CRITIC_TRAIN_STEPS = 5
BATCH_SIZE = 64 * 5
MINIBATCH_SIZE = 64
LAMBDA = 10
OPTIMIZER = optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], IMG_SIDE_LENGTH, IMG_SIDE_LENGTH, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
LABELS = sorted(set(train_labels))
# split the whole dataset into 10 subdatasets for each label
train_images_split = [train_images[train_labels==i] for i in LABELS]
TRAIN_DATASETS = [tf.data.Dataset.from_tensor_slices(imgs).shuffle(imgs.shape[0]).batch(BATCH_SIZE) for imgs in train_images_split]

def get_train_step():
    @tf.function
    def train_step(generator, critic, images):

        G, C = generator, critic
        C_loss, G_loss = 0, 0
        
        ## train only Critic for n steps
        for k in range(CRITIC_TRAIN_STEPS):
            
            # sample noise
            noise = tf.random.normal([MINIBATCH_SIZE, NOISE_DIM])
            # random number
            eps = tf.random.uniform(shape=[1], minval=0., maxval=1.)
            
            with tf.GradientTape() as critic_tape:
                # generated images
                gen_images = G(noise, training=False)
                # real images
                real_images = images[MINIBATCH_SIZE*k : MINIBATCH_SIZE*(k+1)]
                # gradient penalty
                pen_x = eps * real_images + (1 - eps) * gen_images
                grad_pen = tf.gradients(C(pen_x), [pen_x])[0]
                grad_pen = LAMBDA * tf.reduce_mean(tf.square(tf.norm(grad_pen, ord='euclidean') - 1))

                # Critic scores
                real_scores = C(real_images, training=True)
                gen_scores = C(gen_images, training=True)

                # calculate critic loss
                C_loss = critic_loss(real_scores, gen_scores, grad_pen)

            # calculate critic gradients
            C_gradients = critic_tape.gradient(C_loss, C.trainable_variables)
            # update critic weights
            OPTIMIZER.apply_gradients(zip(C_gradients, C.trainable_variables))
           
        ## train only G for 1 step
        # sample noise
        noise = tf.random.normal([MINIBATCH_SIZE, NOISE_DIM])
        
        with tf.GradientTape() as gen_tape:
            # turn noise into generated images
            gen_images = G(noise, training=True)
            # Critic scores
            gen_scores = C(gen_images, training=False)
            # calculate Generator loss
            G_loss = generator_loss(gen_scores)
            
        # calculate Generator gradients 
        G_gradients = gen_tape.gradient(G_loss, G.trainable_variables)
        # update Generator weights
        OPTIMIZER.apply_gradients(zip(G_gradients, G.trainable_variables))

        return C_loss, G_loss
    
    return train_step


def train(epochs, load_models=False, save_images=False, save_models=False):
    """
    load_models : boolean -------- use saved models ?
    epochs : integer -------- number of epochs
    save_images : boolean -------- save generated images ?
    save_models : boolean -------- save models ?
    """
    # load models from files if needed
    if load_models:
        try:
            models = [(tf.keras.models.load_model(f"models/latest/{l}/critic"), tf.keras.models.load_model(f"models/latest/{l}/generator")) for l in LABELS]
        except:
            raise SystemExit("####### Failed to load models ########")
            
    else:
        models = [WGAN_GP().models for _ in LABELS]
    
    C_losses_all, G_losses_all = [], []

    basename = datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S")
    
    # create a dir for saving generated images
    img_save_paths = ["" for _ in range(len(LABELS))]
    if save_images:
        img_save_dir = f"generated/{basename}"
        os.mkdir(img_save_dir)
        img_save_paths = [f"{img_save_dir}/{l}" for l in LABELS]
        for p in img_save_paths:
            os.mkdir(p)

    save_model_every = max(1, int(epochs / 5))
    # create a dir to save models
    models_save_paths = ["" for _ in range(len(LABELS))]
    if save_models:
        models_save_dir = f"models/{basename}"
        os.mkdir(models_save_dir)
        models_save_paths = [f"{models_save_dir}/{l}" for l in LABELS]
        for p in models_save_paths:
            os.mkdir(p)

    for m, dataset, isp, msp, digit in zip(models, TRAIN_DATASETS, img_save_paths, models_save_paths, LABELS): 
        print(f"----------- TRAINING TO GENERATE **{digit}** -----------")
        # there is a need to reinitialize a function every time a new model is trained
        train_step_func = get_train_step()
        critic, generator = m
        C_losses, G_losses = [], []
        pbar = tqdm(total=100)
        
        # training
        for ep in range(epochs):
            start = time.time()
            
            for images_batch in dataset:
                if images_batch.shape[0] != BATCH_SIZE:
                    continue
                # train models on current batch
                C_loss, G_loss = train_step_func(generator, critic, images_batch)
                            
            C_losses.append(C_loss)
            G_losses.append(G_loss)
            
            pbar.update(round(1/epochs * 100, 2))
            
            print(f"DIGIT: {digit}")
            print(f"EPOCH {ep+1}: {time.time() - start:.5f} sec")
            print(f"Critic loss: {C_losses[-1]:.5f}")
            print(f"Generator loss: {G_losses[-1]:.5f}\n")
            
            ## save generated image every ep
            if save_images:
                noise = tf.random.normal([1, NOISE_DIM])
                imgts = generator(noise, training=False).numpy().reshape(IMG_SIDE_LENGTH, IMG_SIDE_LENGTH, 1)
                imgts = (imgts * 127.5) + 127.5
                save_img(f"{isp}/{ep}.jpg", imgts)
                
            ## save model from time to time
            if save_models and ep % save_model_every == 0:
                generator.save(f"{msp}/generator")
                critic.save(f"{msp}/critic")
                
        pbar.close()
        
    return C_losses, G_losses, critic, generator, basename

def usage():
    print("train(epochs, load_models, save_images, save_models)")
    print("returns: C_losses, G_losses, critic, generator, basename")
