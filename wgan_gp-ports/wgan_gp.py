import datetime, os, time, copy
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, initializers, constraints, optimizers, losses
from tensorflow.keras.preprocessing.image import save_img
from helper_funcs import read_dir

DTYPE = tf.dtypes.float16
DTYPE_STR = 'float16'

tf.keras.backend.set_floatx(DTYPE_STR)
#KERNEL_INIT = initializers.RandomNormal(mean=1., stddev=0.02)
KERNEL_INIT = None
IMG_SIDE_LENGTH = 64
NOISE_DIM = 128


class WGAN_GP:
    
    def __init__(self):
        self.models = (self.__build_critic(), self.__build_generator())        

    def __build_critic(self):
        C = models.Sequential(name="Critic")
        
        # 64 x 64 x 3
        C.add(layers.Conv2D(
            64, (3, 3), strides=(1, 1), input_shape=(IMG_SIDE_LENGTH, IMG_SIDE_LENGTH, 3), padding='same', kernel_initializer=KERNEL_INIT
        ))
        # 64 x 64 x 64
        C.add(layers.LeakyReLU(0.2))
        C.add(layers.Conv2D(
            64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=KERNEL_INIT
        ))
        # 32 x 32 x 64
        C.add(layers.LeakyReLU(0.2))
        C.add(layers.Conv2D(
            128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=KERNEL_INIT
        ))
        # 16 x 16 x 128 
        C.add(layers.LeakyReLU(0.2))
        C.add(layers.Conv2D(
            256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=KERNEL_INIT
        ))
        # 8 x 8 x 256
        C.add(layers.Flatten())
        # 8*8*256
        C.add(layers.Dense(1))
        # Output linear
        
        return C

    def __build_generator(self):
        G = models.Sequential(name='Generator')

        # in: 128 x 1
        G.add(layers.Dense(8*8*256, input_shape=(NOISE_DIM,), kernel_initializer=KERNEL_INIT))
        # 8*8*256
        G.add(layers.Reshape((8, 8, 256)))
        # 8 x 8 x 256
        G.add(layers.BatchNormalization())
        G.add(layers.ReLU())
        G.add(layers.Conv2DTranspose(
            256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=KERNEL_INIT
        ))
        # 16 x 16 x 256
        G.add(layers.BatchNormalization())
        G.add(layers.ReLU())
        G.add(layers.Conv2DTranspose(
            128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=KERNEL_INIT
        ))
        # 32 x 32 x 128
        G.add(layers.BatchNormalization())
        G.add(layers.ReLU())
        G.add(layers.Conv2DTranspose(
            64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=KERNEL_INIT
        ))
        # 64 x 64 x 64
        G.add(layers.BatchNormalization())
        G.add(layers.ReLU())
        G.add(layers.Conv2DTranspose(
            3, (3, 3), strides=(1, 1), padding='same', kernel_initializer=KERNEL_INIT, activation='tanh'
        ))
        # 64 x 64 x 3
       
        return G 

def generator_loss(gen_scores):
    return -1 * tf.reduce_mean(gen_scores)

def critic_loss(real_scores, gen_scores, grad_penalty):
    return tf.reduce_mean(gen_scores) - tf.reduce_mean(real_scores) + grad_penalty

CRITIC_TRAIN_STEPS = 5
BATCH_SIZE = 64 * 5
MINIBATCH_SIZE = 64 
LAMBDA = 10
G_OPTIMIZER = optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
C_OPTIMIZER = optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)

@tf.function
def train_step(generator, critic, images):
    
    C, G = critic, generator
    C_loss, G_loss = 0, 0
    
    ## train only Critic for n steps
    for k in range(CRITIC_TRAIN_STEPS):
        
        # sample noise
        noise = tf.random.normal([MINIBATCH_SIZE, NOISE_DIM], dtype=DTYPE)
        # random number
        eps = tf.random.uniform(shape=[1], minval=0., maxval=1., dtype=DTYPE)
        
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
        C_OPTIMIZER.apply_gradients(zip(C_gradients, C.trainable_variables))
       
    ## train only G for 1 step
    # sample noise
    noise = tf.random.normal([MINIBATCH_SIZE, NOISE_DIM], dtype=DTYPE)
    
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
    G_OPTIMIZER.apply_gradients(zip(G_gradients, G.trainable_variables))

    return C_loss, G_loss

def train(epochs, dir_path, load_models=False, save_images=False, save_models=False):
    """
    dir_path : string       -------- path to a directory with .npy files
    load_models : boolean   -------- use saved models ?
    epochs : integer        -------- number of epochs
    save_images : boolean   -------- save generated images ?
    save_models : boolean   -------- save models ?
    """
    basename = datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S")
    critic, generator = WGAN_GP().models
    
    save_model_every = min(max(1, int(epochs / 20)), 15)
    
    checkpoint_dir, checkpoint_prefix, checkpoint = None, None, None
    if save_models:
        checkpoint_dir = f"./training_checkpoints/{basename}"
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(G_OPTIMIZER=G_OPTIMIZER, C_OPTIMIZER=C_OPTIMIZER,
                                            generator=generator, critic=critic)

    # load models from files if needed
    if load_models:
        try:
            checkpoint.restore(tf.train.latest_checkpoint("./training_checkpoints/latest"))
            #generator = tf.keras.models.load_model('models/latest/generator', compile=False)
            #critic = tf.keras.models.load_model('models/latest/critic', compile=False)
        except:
            raise SystemExit("####### Failed to load models ########")

    C_losses, G_losses = [], []

    # create a dir for saving generated images
    if save_images:
        img_save_path = f"generated/{basename}"
        os.mkdir(img_save_path)
        
    # form a list of .npy files' paths
    np_paths = read_dir(dir_path, ".npy")

    pbar = tqdm(total=100)
    # training
    for ep in range(epochs):
        start = time.time()

        np.random.shuffle(np_paths)
        c = 0
        
        while c < len(np_paths):
            # load batch
            real_images = np.load(np_paths[c]).astype(DTYPE_STR)
            # shuffle images inside the batch
            np.random.shuffle(real_images)
            # normalize images to [-1, 1]
            real_images = (real_images - 127.5) / 127.5  
            # train models on current batch
            C_loss, G_loss = train_step(generator, critic, real_images)

            c += 1
                       
        C_losses.append(C_loss)
        G_losses.append(G_loss)
        
        pbar.update(round(1/epochs * 100, 2))
        
        print(f"EP {ep+1}: {time.time() - start:.1f}s ----- C loss: {C_losses[-1]:.2f} ----- G loss: {G_losses[-1]:.2f}")
        
        ## save generated image every ep
        if save_images:
            noise = tf.random.normal([1, NOISE_DIM], dtype=DTYPE)
            imgts = generator(noise, training=False).numpy().reshape(IMG_SIDE_LENGTH, IMG_SIDE_LENGTH, 3)
            # [-1, 1] ---> [0, 255]
            imgts = (imgts * 127.5) + 127.5
            save_img(f"{img_save_path}/{ep}.jpg", imgts)
            
        ## save model from time to time
        if save_models and (ep % save_model_every == 0 or ep == epochs - 1):
            checkpoint.save(file_prefix=checkpoint_prefix)
#            generator.save(f"{models_save_path}/generator")
 #           critic.save(f"{models_save_path}/critic")

            
    pbar.close()
    
    return C_losses, G_losses, critic, generator, basename

def usage():
    print("train(epochs, dir_path, load_models, save_images, save_models)")
    print("returns: C_losses, G_losses, critic, generator, basename")
