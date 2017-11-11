from os.path import join as ojoin

from keras.preprocessing.image import *
from PIL import Image
from src.data.coco_video import COCOPriMatrix


class VideoDataGenerator(ImageDataGenerator):
    def random_transform(self, x, seed=None):
        """Randomly augment a single image tensor.
        # Arguments
            x: 4D tensor, single image.
            seed: random seed.
        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis
        img_col_axis = self.col_axis
        img_channel_axis = self.channel_axis

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range,
                                                    self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.
                                   height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.
                                   width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1],
                                       2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta),
                                         np.cos(theta), 0], [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(
                transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0], [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(
                transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(
                transform_matrix, zoom_matrix)

        horizontal_flip = False
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                horizontal_flip = True

        vertical_flip = False
        if self.vertical_flip:
            if np.random.random() < 0.5:
                vertical_flip = True

        if transform_matrix is not None:
            h, w = x[0].shape[img_row_axis], x[0].shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(
                transform_matrix, h, w)

        arr = []
        for i in range(x.shape[0]):
            u = x[i]
            if transform_matrix is not None:
                u = apply_transform(
                    u,
                    transform_matrix,
                    img_channel_axis - 1,
                    fill_mode=self.fill_mode,
                    cval=self.cval)

            if self.channel_shift_range != 0:
                u = random_channel_shift(u, self.channel_shift_range,
                                         img_channel_axis)
            if horizontal_flip:
                u = flip_axis(u, img_col_axis - 1)

            if vertical_flip:
                u = flip_axis(u, img_row_axis - 1)
            arr.append(u)
        x = np.stack(arr)

        return x

    def standardize(self, u):
        """Apply the normalization configuration to a batch of inputs.
        # Arguments
            x: batch of inputs to be normalized.
        # Returns
            The inputs, normalized.
        """
        arr = []

        for i in range(u.shape[0]):
            x = u[i]
            if self.rescale:
                x *= self.rescale
            if self.samplewise_center:
                x -= np.mean(x, keepdims=True)
            if self.samplewise_std_normalization:
                x /= np.std(x, keepdims=True) + 1e-7

            if self.featurewise_center:
                if self.mean is not None:
                    x -= self.mean
                else:
                    warnings.warn('This ImageDataGenerator specifies '
                                  '`featurewise_center`, but it hasn\'t'
                                  'been fit on any training data. Fit it '
                                  'first by calling `.fit(numpy_data)`.')
            if self.featurewise_std_normalization:
                if self.std is not None:
                    x /= (self.std + 1e-7)
                else:
                    warnings.warn(
                        'This ImageDataGenerator specifies '
                        '`featurewise_std_normalization`, but it hasn\'t'
                        'been fit on any training data. Fit it '
                        'first by calling `.fit(numpy_data)`.')
            if self.zca_whitening:
                if self.principal_components is not None:
                    flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                    whitex = np.dot(flatx, self.principal_components)
                    x = np.reshape(whitex, x.shape)
                else:
                    warnings.warn('This ImageDataGenerator specifies '
                                  '`zca_whitening`, but it hasn\'t'
                                  'been fit on any training data. Fit it '
                                  'first by calling `.fit(numpy_data)`.')
            arr.append(x)
        x = np.stack(arr)
        return x

    def flow_from_directory(self,
                            directory,
                            split,
                            target_size,
                            batch_size=32,
                            shuffle=True,
                            seed=None):
        return VideoDataIterator(
            directory,
            self,
            split,
            target_size=target_size,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed)


class VideoDataIterator(Iterator):
    def __init__(self,
                 directory,
                 data_generator,
                 split,
                 target_size=(16, 64, 64, 3),
                 batch_size=32,
                 shuffle=True,
                 seed=None):

        self.directory = directory
        self.data_generator = data_generator
        self.data = COCOPriMatrix(
            ojoin(self.directory, 'annotations',
                  'instances_{}.json'.format(split)))
        self.filenames = self.data.getImgIds()
        self.nb_sample = len(self.filenames)
        self.num_classes = len(self.data.cats)
        self.target_size = target_size
        self.num_frames = self.data.fdata['data'].shape[1]
        self.width = self.data.fdata['data'].shape[2]
        self.height = self.data.fdata['data'].shape[3]

        super(VideoDataIterator, self).__init__(self.nb_sample, batch_size,
                                                shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)

        # The transformation of images is not under thread lock so it can be
        # done in parallel
        batch_x = np.zeros((current_batch_size, ) + self.target_size)
        batch_y = np.zeros((current_batch_size, ) + (len(self.data.cats), ))

        # build batch of image data
        for i, j in enumerate(index_array):
            video = self.data.fdata['data'][j]
            if video.shape[0] != self.target_size[0]:
                video = video[:self.target_size[0]]
            if video.shape[1:] != self.target_size[1:]:
                video = np.stack([
                    np.array(
                        Image.fromarray(video[i].astype('uint8')).resize(
                            self.target_size[1:-1]))
                    for i in range(len(video))
                ])

            video = self.data_generator.random_transform(video)
            video = self.data_generator.standardize(video)

            labels = self.data.fdata['labels'][j]

            batch_x[i] = video
            batch_y[i] = labels

        return batch_x, batch_y
