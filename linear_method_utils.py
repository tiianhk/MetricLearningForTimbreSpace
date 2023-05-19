from sklearn.metrics import classification_report, \
                            v_measure_score, \
                            confusion_matrix, \
                            ConfusionMatrixDisplay, \
                            pairwise_distances
from sklearn.neighbors import NearestCentroid
from instrument_name_utils import INST_NAME_TO_INST_FAM_NAME_DICT
import matplotlib.pyplot as plt
from visual import visualizer

class linear_metric_learner():

    def __init__(self, clf_inst, clf_inst_fam, X_train, y_train, X_test, y_test, transform=True):
        """
        Args:
            clf_inst/clf_inst_fam: a classifier or a pipeline in which the last one is a classifier
            X_train/X_test: features, shape: (n_samples, n_features)
            y_train/y_test: instrument labels, shape: (n_features,)
        """
        
        # classifiers for instrument / instrument family
        self.clf_inst = clf_inst
        self.clf_inst_fam = clf_inst_fam
        
        # X and y
        self.X_train = X_train
        self.y_train_inst = y_train
        self.X_test = X_test
        self.y_test_inst = y_test

        # whether to do transformation on X before the classification
        self.transform = transform
        
        # utils for label mapping
        self.inst_list = list(INST_NAME_TO_INST_FAM_NAME_DICT.keys())
        self.inst_fam_list = list(INST_NAME_TO_INST_FAM_NAME_DICT.values())
        self.inst_fam_id = {
            'string': 0,
            'woodwind': 1,
            'brass': 2
        }
        
        # instrument family labels
        self.y_train_inst_fam = [self.inst_fam_id[self.inst_fam_list[i]] for i in self.y_train_inst]
        self.y_test_inst_fam = [self.inst_fam_id[self.inst_fam_list[i]] for i in self.y_test_inst]

    def fit(self):
        
        # fitting on the training data
        self.clf_inst.fit(self.X_train, y=self.y_train_inst)
        self.clf_inst_fam.fit(self.X_train, y=self.y_train_inst_fam)
        
        # inference on the test data
        self.y_pred_inst = self.clf_inst.predict(self.X_test)
        self.y_pred_inst_fam = self.clf_inst_fam.predict(self.X_test)

    def check_fit(self):
        if not hasattr(self, 'y_pred_inst'):
            print('fitting first..\n')
            self.fit()

    def evaluate(self, plot_cm=True):
        self.check_fit()

        # v_measure
        v_measure_inst = v_measure_score(self.y_test_inst, self.y_pred_inst)
        print(f'v_measure for instrument clustering: {v_measure_inst:.3f}')
        v_measure_inst_fam = v_measure_score(self.y_test_inst_fam, self.y_pred_inst_fam)
        print(f'v_measure for instrument family clustering: {v_measure_inst_fam:.3f}\n')
        
        # classification report
        report_inst = classification_report(self.y_test_inst, self.y_pred_inst, target_names=self.inst_list)
        print('classification report for instrument:\n', report_inst)
        report_inst_fam = classification_report(self.y_test_inst_fam, self.y_pred_inst_fam, 
                                                target_names=list(self.inst_fam_id.keys()))
        print('classification report for instrument family:\n', report_inst_fam)
        
        # confusion matrix
        if plot_cm:
            print('confusion matrix for instrument classification:')
            fig, ax1 = plt.subplots(figsize=(10,10))
            cm = confusion_matrix(self.y_test_inst, self.y_pred_inst, normalize='true')
            cmd = ConfusionMatrixDisplay(cm, display_labels=self.inst_list)
            cmd.plot(xticks_rotation='vertical', ax=ax1)
            plt.show()
            print('confusion matrix for instrument family classification:')
            fig, ax2 = plt.subplots(figsize=(4,4))
            cm = confusion_matrix(self.y_test_inst_fam, self.y_pred_inst_fam, normalize='true')
            cmd = ConfusionMatrixDisplay(cm, display_labels=list(self.inst_fam_id.keys()))
            cmd.plot(ax=ax2)
            plt.show()

    def visualize_train_data(self, n_components=2, method='umap'):
        self.check_fit()
        
        # prepare data
        print('visualizing the training data..')
        if self.transform:
            print('data is firstly transformed based on the instrument pipeline..')
            x = self.clf_inst[:-1].transform(self.X_train)
        else:
            x = self.X_train
        inst = [self.inst_list[i] for i in self.y_train_inst]
        
        # visualize
        viz = visualizer(x, inst, method=method, n_components=n_components)
        viz.generate().show()
    
    def get_centroids(self, out_dir=None):
        self.check_fit()

        print('extracting embedding centroids for each instrument category..')
        if self.transform:
            assert isinstance(self.clf_inst[-1], NearestCentroid)
            self.centroids_inst = self.clf_inst[-1].centroids_
        else:
            assert isinstance(self.clf_inst, NearestCentroid)
            self.centroids_inst = self.clf_inst.centroids_
        
        # save as file
        if out_dir is not None:
            # to-do: store the centroids as a file
            pass

    def plot_centroid_distances(self):
        if not hasattr(self, 'centroids_inst'):
            self.get_centroids()
        print('Pairwise distance matrix for instrument centroids:')
        Dist = pairwise_distances(self.centroids_inst)
        plt.figure(figsize=(6,6))
        plt.imshow(Dist, origin='lower')
        plt.colorbar()
        plt.xticks(range(len(self.inst_list)), self.inst_list, rotation='vertical')
        plt.yticks(range(len(self.inst_list)), self.inst_list)
        plt.show()

