import gin
import os
import numpy as np
import tensorflow as tf
import librosa
import random
import openl3
from tqdm import tqdm
from tensorflow.keras.utils import Sequence
from instrument_name_utils import INST_NAME_TO_INST_FAM_NAME_DICT
from paras import RNG_STATE
from sklearn.preprocessing import normalize


CREPE_SAMPLE_RATE = 16000
CREPE_FRAME_SIZE = 1024
_AUTOTUNE = tf.data.experimental.AUTOTUNE


'''use same train/test data as midi-ddsp's synthesis generator.'''


# https://github.com/magenta/ddsp/blob/main/ddsp/training/data.py
# https://github.com/magenta/midi-ddsp/blob/main/midi_ddsp/data_handling/get_dataset.py


def get_framed_lengths(input_length, frame_size, hop_size, padding='center'):
    """Give a strided framing, such as tf.signal.frame, gives output lengths.
    Args:
    input_length: Original length along the dimension to be framed.
    frame_size: Size of frames for striding.
    hop_size: Striding, space between frames.
    padding: Type of padding to apply, ['valid', 'same', 'center']. 'valid' is
      a no-op. 'same' applies padding to the end such that
      n_frames = n_t / hop_size. 'center' applies padding to both ends such that
      each frame timestamp is centered and n_frames = n_t / hop_size + 1.
    Returns:
    n_frames: Number of frames left after striding.
    padded_length: Length of the padded signal before striding.
    """
    # Use numpy since this function isn't used dynamically.
    def get_n_frames(length):
        return int(np.floor((length - frame_size) / hop_size)) + 1

    if padding == 'valid':
        padded_length = input_length
        n_frames = get_n_frames(input_length)

    elif padding == 'center':
        padded_length = input_length + frame_size
        n_frames = get_n_frames(padded_length)

    elif padding == 'same':
        n_frames = int(np.ceil(input_length / hop_size))
        padded_length = (n_frames - 1) * hop_size + frame_size

    return n_frames, padded_length


class DataProvider(object):
    """Base class for returning a dataset."""

    def __init__(self, sample_rate, frame_rate):
        """DataProvider constructor.
        Args:
          sample_rate: Sample rate of audio in the dataset.
          frame_rate: Frame rate of features in the dataset.
        """
        self._sample_rate = sample_rate
        self._frame_rate = frame_rate

    @property
    def sample_rate(self):
        """Return dataset sample rate, must be defined in the constructor."""
        return self._sample_rate

    @property
    def frame_rate(self):
        """Return dataset feature frame rate, must be defined in the constructor."""
        return self._frame_rate

    def get_dataset(self, shuffle):
        """A method that returns a tf.data.Dataset."""
        raise NotImplementedError

    def get_batch(self,
                  batch_size,
                  shuffle=True,
                  repeats=-1,
                  drop_remainder=True):
        """Read dataset.
        Args:
          batch_size: Size of batch.
          shuffle: Whether to shuffle the examples.
          repeats: Number of times to repeat dataset. -1 for endless repeats.
          drop_remainder: Whether the last batch should be dropped.
        Returns:
          A batched tf.data.Dataset.
        """
        dataset = self.get_dataset(shuffle)
        dataset = dataset.repeat(repeats)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(buffer_size=_AUTOTUNE)
        return dataset


@gin.register
class TFRecordProvider(DataProvider):
    """Class for reading TFRecords and returning a dataset."""

    def __init__(self,
                 file_pattern=None,
                 example_secs=4,
                 sample_rate=16000,
                 frame_rate=250,
                 centered=False):
        """RecordProvider constructor."""
        super().__init__(sample_rate, frame_rate)
        self._file_pattern = file_pattern or self.default_file_pattern
        self._audio_length = example_secs * sample_rate
        self._audio_16k_length = example_secs * CREPE_SAMPLE_RATE
        self._feature_length = self.get_feature_length(centered)

    def get_feature_length(self, centered):
        """Take into account center padding to get number of frames."""
        # Number of frames is independent of frame size for "center/same" padding.
        hop_size = CREPE_SAMPLE_RATE / self.frame_rate
        padding = 'center' if centered else 'same'
        return get_framed_lengths(
            self._audio_16k_length, CREPE_FRAME_SIZE, hop_size, padding)[0]

    @property
    def default_file_pattern(self):
        """Used if file_pattern is not provided to constructor."""
        raise NotImplementedError(
            'You must pass a "file_pattern" argument to the constructor or '
            'choose a FileDataProvider with a default_file_pattern.')

    def get_dataset(self, shuffle=True):
        """Read dataset.
        Args:
          shuffle: Whether to shuffle the files.
        Returns:
          dataset: A tf.dataset that reads from the TFRecord.
        """
        def parse_tfexample(record):
            return tf.io.parse_single_example(record, self.features_dict)

        filenames = tf.data.Dataset.list_files(self._file_pattern, shuffle=shuffle)
        dataset = filenames.interleave(
            map_func=tf.data.TFRecordDataset,
            cycle_length=40,
            num_parallel_calls=_AUTOTUNE)
        dataset = dataset.map(parse_tfexample, num_parallel_calls=_AUTOTUNE)
        return dataset

    @property
    def features_dict(self):
        """Dictionary of features to read from dataset."""
        return {
            'audio':
                tf.io.FixedLenFeature([self._audio_length], dtype=tf.float32),
            'audio_16k':
                tf.io.FixedLenFeature([self._audio_16k_length], dtype=tf.float32),
            'f0_hz':
                tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
            'f0_confidence':
                tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
            'loudness_db':
                tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
        }


@gin.register
class LegacyTFRecordProvider(TFRecordProvider):
    """Class for reading TFRecords and returning a dataset."""

    @property
    def features_dict(self):
        """Dictionary of features to read from dataset."""
        return {
            'audio':
                tf.io.FixedLenFeature([self._audio_length], dtype=tf.float32),
            'f0_hz':
                tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
            'f0_confidence':
                tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
            'loudness_db':
                tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
        }


@gin.register
class Urmp(LegacyTFRecordProvider):
    """Urmp training set."""

    def __init__(self,
                 base_dir,
                 instrument_key='tpt',
                 split='train',
                 suffix=None):
        """URMP dataset for either a specific instrument or all instruments.
        Args:
          base_dir: Base directory to URMP TFRecords.
          instrument_key: Determines which instrument to return. Choices include
            ['all', 'bn', 'cl', 'db', 'fl', 'hn', 'ob', 'sax', 'tba', 'tbn',
            'tpt', 'va', 'vc', 'vn'].
          split: Choices include ['train', 'test'].
          suffix: Choices include [None, 'batched', 'unbatched'], but broadly
            applies to any suffix adding to the file pattern.
            When suffix is not None, will add "_{suffix}" to the file pattern.
            This option is used in gs://magentadata/datasets/urmp/urmp_20210324.
            With the "batched" suffix, the dataloader will load tfrecords
            containing segmented audio samples in 4 seconds. With the "unbatched"
            suffix, the dataloader will load tfrecords containing unsegmented
            samples which could be used for learning note sequence in URMP dataset.
        """
        self.instrument_key = instrument_key
        self.split = split
        self.base_dir = base_dir
        self.suffix = '' if suffix is None else '_' + suffix
        super().__init__()

    @property
    def default_file_pattern(self):
        if self.instrument_key == 'all':
            file_pattern = 'all_instruments_{}{}.tfrecord*'.format(
                self.split, self.suffix)
        else:
            file_pattern = 'urmp_{}_solo_ddsp_conditioning_{}{}.tfrecord*'.format(
                self.instrument_key, self.split, self.suffix)

        return os.path.join(self.base_dir, file_pattern)


@gin.register
class UrmpMidi(Urmp):
    """Urmp training set with midi note data.
    This class loads the segmented data in tfrecord that contains 4-second audio
    clips of the URMP dataset. To load tfrecord that contains unsegmented full
    piece of URMP recording, please use `UrmpMidiUnsegmented` class instead.
    """

    _INSTRUMENTS = ['vn', 'va', 'vc', 'db', 'fl', 'ob', 'cl', 'sax', 'bn', 'tpt',
                    'hn', 'tbn', 'tba']

    @property
    def features_dict(self):
        base_features = super().features_dict
        base_features.update({
            'note_active_velocities':
                tf.io.FixedLenFeature([self._feature_length * 128], tf.float32),
            'note_active_frame_indices':
                tf.io.FixedLenFeature([self._feature_length * 128], tf.float32),
            'instrument_id': tf.io.FixedLenFeature([], tf.string),
            'recording_id': tf.io.FixedLenFeature([], tf.string),
            'power_db':
                tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
            'note_onsets':
                tf.io.FixedLenFeature([self._feature_length * 128],
                                      dtype=tf.float32),
            'note_offsets':
                tf.io.FixedLenFeature([self._feature_length * 128],
                                      dtype=tf.float32),
        })
        return base_features

    def get_dataset(self, shuffle=True):

        instrument_ids = range(len(self._INSTRUMENTS))
        inst_vocab = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(self._INSTRUMENTS, instrument_ids),
            -1)

        def _reshape_tensors(data):
            data['note_active_frame_indices'] = tf.reshape(
                data['note_active_frame_indices'], (-1, 128))
            data['note_active_velocities'] = tf.reshape(
                data['note_active_velocities'], (-1, 128))
            data['instrument_id'] = inst_vocab.lookup(data['instrument_id'])
            data['midi'] = tf.argmax(data['note_active_frame_indices'], axis=-1)
            data['f0_hz'] = data['f0_hz'][..., tf.newaxis]
            data['loudness_db'] = data['loudness_db'][..., tf.newaxis]
            onsets = tf.reduce_sum(
                tf.reshape(data['note_onsets'], (-1, 128)), axis=-1)
            data['onsets'] = tf.cast(onsets > 0, tf.int64)
            offsets = tf.reduce_sum(
                tf.reshape(data['note_offsets'], (-1, 128)), axis=-1)
            data['offsets'] = tf.cast(offsets > 0, tf.int64)

            return data

        ds = super().get_dataset(shuffle)
        ds = ds.map(_reshape_tensors, num_parallel_calls=_AUTOTUNE)
        return ds


def get_tfrecord_length(dataloader):
    """Return the length of a dataloader."""
    dataset_single = dataloader.get_batch(batch_size=1,
                                          shuffle=False,
                                          repeats=1)
    c = 0
    for _ in dataset_single:
        c += 1
    return c


def get_tfdata(data_dir, split):
    """Return dataloader based on hyperparameters."""

    data_loader = UrmpMidi(data_dir, instrument_key='all',
                           split=split, suffix='batched')
    data = data_loader.get_batch(batch_size=1, shuffle=False,
                                    repeats=1, drop_remainder=False)

    length = get_tfrecord_length(data_loader)

    return iter(data), length


def get_dataset(data_dir, split, duration):

    dataset = []
    data_iter, length = get_tfdata(data_dir, split)

    for i in range(length):
        data = next(data_iter)
        example = {}
        audio = data['audio'].numpy()[0]
        example['duration'] = duration
        example['instrument_id'] = data['instrument_id'].numpy()[0]
        example['recording_id'] = data['recording_id'].numpy()[0].decode('utf-8')
        if duration == 4.0:
            example['audio'] = audio
            dataset.append(example)
        elif duration == 1.0:
            sample_rate = 16000
            hopsize = int(0.5 * sample_rate)
            st = 0
            en = sample_rate
            while st < int(0.5 * len(audio)):
                example['audio'] = audio[st:en]
                dataset.append(example)
                st += hopsize
                en += hopsize
        else:
            raise ValueError('duration should be 4.0 or 1.0')

    return dataset


def get_mfcc_and_label(dataset, sr=CREPE_SAMPLE_RATE):
    """retrieve time averaged mfcc and instrument id."""
    X, y = [], []
    for example in tqdm(dataset):
        mfcc = librosa.feature.mfcc(example['audio'], sr)
        mfcc_ave = np.mean(mfcc, axis=-1)
        X.append(mfcc_ave)
        y.append(example['instrument_id'])
    return np.array(X), np.array(y)


def get_melspec_and_label(dataset, sr=CREPE_SAMPLE_RATE):
    """retrieve melspectrogram, instrument name and instrument family name."""
    melspec_and_label = []
    duration = dataset[0]['duration']
    inst_list = list(INST_NAME_TO_INST_FAM_NAME_DICT.keys())
    for example in dataset:
        if duration == 4.0:
            melspec = librosa.feature.melspectrogram(
                y=example['audio'], sr=sr, power=1)[...,np.newaxis]
        elif duration == 1.0:
            melspec = librosa.feature.melspectrogram(
                y=example['audio'], sr=sr, win_length=512,
                hop_length=128, power=1)[...,np.newaxis]
        inst = inst_list[example['instrument_id']]
        inst_fam = INST_NAME_TO_INST_FAM_NAME_DICT[inst]
        melspec_and_label.append({'melspec': melspec, 'inst': inst, 'inst_fam': inst_fam})
    return melspec_and_label


class TripletSequence(Sequence):

    def __init__(self, data, n_batch=5000, batch_size=16, hierarchy=True):

        self.data = data
        self.n_batch = n_batch
        self.batch_size = batch_size
        self.hierarchy = hierarchy

        random.seed(RNG_STATE)
        self.inst_list = list(INST_NAME_TO_INST_FAM_NAME_DICT.keys())
        self.indices = [[] for i in range(self.n_batch)]
        for i in range(self.n_batch):
            inst_batch = random.choices(self.inst_list, k=self.batch_size)
            for inst in inst_batch:
                self.indices[i].append(self.generate_samples_for_anchor_inst(inst))
        self.indices = np.array(self.indices)  # shape (n_batch, batch_size, n_examples)

    def __len__(self):
        return self.n_batch

    def __getitem__(self, idx):
        indices = self.indices[idx]
        if self.hierarchy:
            indices_in_fam = indices[:,:3]
            X_a_in_fam, X_p_in_fam, X_n_in_fam = self.get_batched_melspec(indices_in_fam)
            y_in_fam = np.ones([self.batch_size,])  # alpha margin
            indices_out_fam = indices[:,[0,2,3]]
            X_a_out_fam, X_p_out_fam, X_n_out_fam = self.get_batched_melspec(indices_out_fam)
            y_out_fam = np.ones([self.batch_size,]) * 2  # beta margin
            batched_X = {
                'anchor_input': np.concatenate((X_a_in_fam, X_a_out_fam)),
                'positive_input': np.concatenate((X_p_in_fam, X_p_out_fam)),
                'negative_input': np.concatenate((X_n_in_fam, X_n_out_fam))
            }
            batched_y = np.concatenate((y_in_fam, y_out_fam))
        else:
            X_a, X_p, X_n = self.get_batched_melspec(indices)
            batched_X = {
                'anchor_input': X_a,
                'positive_input': X_p,
                'negative_input': X_n
            }
            batched_y = np.ones([self.batch_size,])
        return batched_X, batched_y

    def get_batched_melspec(self, indices):
        """
        Arg:
            indices: np.array, shape (self.batch_size, 3)
        Return:
            anchor_melspec, positive_melspec, negative_melspec:
                np.array, shape (self.batch_size, frequency, time, channels=1)
        """
        anchor_melspec, positive_melspec, negative_melspec = [],[],[]
        for i in range(self.batch_size):
            anchor_melspec.append(self.data[indices[i,0]]['melspec'])
            positive_melspec.append(self.data[indices[i,1]]['melspec'])
            negative_melspec.append(self.data[indices[i,2]]['melspec'])
        return np.array(anchor_melspec), np.array(positive_melspec), np.array(negative_melspec)

    def generate_samples_for_anchor_inst(self, inst):
        """
        Arg:
            inst: str, instrument name for anchor/positive
        Return:
            anchor_id, positive_id, in_fam_negative_id, out_fam_negative_id: int, indices for self.data
        """
        inst_fam = INST_NAME_TO_INST_FAM_NAME_DICT[inst]
        # anchor and positive
        anchor_id = random.choice(self.get_idx_from_category([inst]))
        positive_id = random.choice(self.get_idx_from_category([inst]))
        # negative
        if self.hierarchy:
            # from the same instrument family
            in_fam_negative_inst_list = [i for i in self.inst_list \
                                         if INST_NAME_TO_INST_FAM_NAME_DICT[i] == inst_fam \
                                         and i != inst]
            # outside the instrument family
            out_fam_negative_inst_list = [i for i in self.inst_list \
                                          if INST_NAME_TO_INST_FAM_NAME_DICT[i] != inst_fam]
            in_fam_negative_id = random.choice(self.get_idx_from_category(in_fam_negative_inst_list))
            out_fam_negative_id = random.choice(self.get_idx_from_category(out_fam_negative_inst_list))
            return anchor_id, positive_id, in_fam_negative_id, out_fam_negative_id
        else:
            negative_inst_list = [i for i in self.inst_list if i != inst]
            negative_id = random.choice(self.get_idx_from_category(negative_inst_list))
            return anchor_id, positive_id, negative_id

    def get_idx_from_category(self, category):
        """
        Arg:
            category: a list, instrument category
        Return:
            idx: a list, indices of examples belonging to the category
        """
        # cache
        if not hasattr(self, "category_to_idx"):
            self.category_to_idx = {inst: [] for inst in self.inst_list}
            for i, example in enumerate(self.data):
                self.category_to_idx[example['inst']].append(i)
        idx = []
        for inst in category:
            idx.extend(self.category_to_idx[inst])
        return idx


def get_openl3_and_label(dataset, sr=CREPE_SAMPLE_RATE):
    """retrieve time averaged openl3 and instrument id."""
    try:
        assert dataset[0]['duration'] == 4.0
        if len(dataset) == 9982:
            X = np.load('./cache/openl3_train_dur_4.npy')
        elif len(dataset) == 2498:
            X = np.load('./cache/openl3_test_dur_4.npy')
    except:
        X = []
        model = openl3.models.load_audio_embedding_model(
            input_repr="mel128", content_type="music", embedding_size=512)
        for example in tqdm(dataset):
            emb, _ = openl3.get_audio_embedding(example['audio'], sr, model=model,
                                                center=False, hop_size=0.5, verbose=0)
            X.append(np.mean(emb, axis=0))
        X = np.array(X)
    y = np.array([example['instrument_id'] for example in dataset])
    return X, y


def backbone_inference(backbone, data, normalized=True):
    """get the 64d embeddings using a backbone model."""
    X, y = [], []
    inst_list = list(INST_NAME_TO_INST_FAM_NAME_DICT.keys())
    for item in tqdm(data):
        x = item['melspec'][np.newaxis,...]
        emb = backbone(x).numpy()
        if normalized:
            emb = normalize(emb)
        emb = np.squeeze(emb)
        inst_id = inst_list.index(item['inst'])
        X.append(emb)
        y.append(inst_id)
    return np.array(X), np.array(y)

