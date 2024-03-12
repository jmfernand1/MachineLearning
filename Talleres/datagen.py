#DataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
import numpy as np 

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, df, batch_size = 32, dim=(32,32), n_channels=3,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indices = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Find list of IDs
        df_batch = self.df.iloc[batch_indices,:]
        df_batch = df_batch.reset_index()
        # Generate data
        X, y = self.__data_generation(df_batch)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, df_batch):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i in range(len(df_batch)):
            img = image.load_img(df_batch.loc[i,'path'], target_size=self.dim)
            X[i] = image.img_to_array(img)
            y[i] = df_batch.loc[i,'label']

        return X, y