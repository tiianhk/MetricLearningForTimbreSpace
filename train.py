import os
os.chdir('/scratch/ht2208/metric-learning')
from data import get_dataset, get_melspec_and_label, TripletSequence
from model import create_triplet_model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import argparse

def hierarchical_triplet_hinge_loss(alpha=0.5, beta=1.0):
    def loss_fn(y_true, y_pred):
        """
        Arg:
            y_true: (batch_size,), 1 for alpha, 2 for beta
            y_pred: (batch_size, 2), cosine similarity  
                note, the similarity score is high when items are similar, 
                so negating the score actually gives a distance score. 
                shaped like [[anchor_positive, anchor_negative], ...]
        Return:
            the loss
        """
        if beta is None:
            margin = y_true * alpha
        else:
            margin = tf.where(tf.equal(y_true, 1.), alpha, tf.where(tf.equal(y_true, 2.), beta, y_true))
        anchor_positive = y_pred[:,0]
        anchor_negative = y_pred[:,1]
        loss = K.mean(K.maximum(0., margin - anchor_positive + anchor_negative))
        return loss
    return loss_fn

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='metric learning for timbre space.')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--hierarchy', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    
    print('loading data..')
    train_set = get_dataset(data_dir='../midi-ddsp/data', split='train', duration=4.0)
    print('getting melspec and labels..')
    train_data = get_melspec_and_label(train_set)
    triplet_sequence = TripletSequence(train_data, n_batch=10000,
                                       batch_size=args.batch_size,
                                       hierarchy=args.hierarchy)
    model = create_triplet_model(input_shape=triplet_sequence[0][0]['anchor_input'].shape[1:])
    model.compile(optimizer=Adam(learning_rate=args.learning_rate, clipnorm=1.0),
	          loss=hierarchical_triplet_hinge_loss(alpha=args.alpha, beta=args.beta))

    log_dir = os.path.join('./log/',f'hie_{args.hierarchy}'\
                                    f'_batchsize_{args.batch_size}'\
				    f'_alpha_{args.alpha}'\
                                    f'_beta_{args.beta}'\
				    f'_lr_{args.learning_rate}')
    os.makedirs(log_dir)

    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=1, min_delta=1e-3, min_lr=1e-6)
    csv_log = tf.keras.callbacks.CSVLogger(os.path.join(log_dir,'train.log'))

    print('training starts..')
    model.fit(triplet_sequence, verbose=2, epochs=30, callbacks=[es,reduce_lr,csv_log])
    model.save(os.path.join(log_dir,'model'))

