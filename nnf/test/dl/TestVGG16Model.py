# -*- coding: utf-8 -*-
"""
.. module:: TestVGG16Model
   :platform: Unix, Windows
   :synopsis: Represent TestVGG16Model and related classes.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import numpy as np
import scipy.io
import os

# Local Imports
from nnf.db.NNdb import NNdb
from nnf.db.Format import Format
from nnf.db.Dataset import Dataset
from nnf.db.NNPatch import NNPatch
from nnf.db.DbSlice import DbSlice
from nnf.db.Selection import Select
from nnf.db.Selection import Selection
from nnf.db.preloaded.MnistDb import MnistDb
from nnf.db.preloaded.Cifar10Db import Cifar10Db
from nnf.core.NNCfg import VGG16Cfg
from nnf.core.NNPatchMan import NNPatchMan
from nnf.core.models.VGG16Model import VGG16Model
from nnf.core.models.NNModelPhase import NNModelPhase
from nnf.core.generators.NNPatchGenerator import NNPatchGenerator
from nnf.alg.LDA import LDA
from nnf.alg.Util import Util


class VGG16Patch(NNPatch):
    def generate_nnmodels(self):
        return VGG16Model(callbacks={'predict':TestVGG16Model._fn_predict})

class VGG16PatchGen(NNPatchGenerator):
    def new_nnpatch(self, h, w, offset):
        return VGG16Patch(h, w, offset, True)

class TestVGG16Model(object):
    """TestVGG16Model to test CNN model."""

    ##########################################################################
    # Public Interface
    ##########################################################################
    def Test_preloaded_db(self):    
        cwd = os.getcwd()
        model_folder = os.path.join(cwd, "ModelFolder")

        # Get the file path for mnist database
        # db_file_path = os.path.join(cwd, "DataFolder", "keras", "mnist.npz")

        # Get the file path for Cifar10 database
        db_file_path = os.path.join(cwd, "DataFolder", "keras", "cifar-10-batches-py")

        nnpatchman = NNPatchMan(VGG16PatchGen())
        vggcfg = VGG16Cfg()

        # Utilizing preloaded db
        #vggcfg.preloaded_db = MnistDb(db_file_path, debug=True)
        vggcfg.preloaded_db = Cifar10Db(db_file_path, debug=True)
        vggcfg.weights_dir = model_folder
        nnpatchman.predict(vggcfg)




        cwd = os.getcwd()
        model_folder = os.path.join(cwd, "ModelFolder")

        nnpatchman = NNPatchMan(CNNPatchGen())
        cnncfg = CNNCfg()

        # Get the file path for mnist database
        # db_file_path = os.path.join(cwd, "DataFolder", "keras", "mnist.npz")

        # Get the file path for Cifar10 database
        db_file_path = os.path.join(cwd, "DataFolder", "keras", "cifar-10-batches-py")

        # Utilizing preloaded db
        # cnncfg.preloaded_db = MnistDb(db_file_path, debug=True)
        cnncfg.preloaded_db = Cifar10Db(db_file_path, debug=True)

        # To save the model & weights
        # cnncfg.model_dir = model_folder

        # To save the weights only      
        # cnncfg.weights_dir = model_folder

        cnncfg.numepochs = 5
        cnncfg.nb_val_samples = 8 #800
        cnncfg.steps_per_epoch = 5 #600

        nnpatchman.train(cnncfg)
        #nnpatchman.test(cnncfg)
        nnpatchman.predict(cnncfg)

    def Test(self):
        # Get the current working directory, define a `DataFolder`
        cwd = os.getcwd()
        data_folder = os.path.join(cwd, "DataFolder")
        model_folder = os.path.join(cwd, "ModelFolder")

        # Load image database `AR`
        matStruct = scipy.io.loadmat(os.path.join(data_folder, 'IMDB_66_66_AR_8.mat'),
                                    struct_as_record=False, squeeze_me=True)
        imdb_obj = matStruct['imdb_obj']

        # Training, Validation, Testing databases
        nndb = NNdb('original', imdb_obj.db, 8, True)
        sel = Selection()
        sel.use_rgb = True
        sel.histeq = True
        sel.scale = (224, 224) #(150, 150)  #(100, 100) #(224, 224)
        #sel.tr_col_indices = np.uint8(np.array([0, 1, 2, 3, 4, 5]))
        #sel.val_col_indices = np.uint8(np.array([6]))
        #sel.te_col_indices = np.uint8(np.array([7]))
        sel.te_col_indices = np.uint8(np.array([0, 1, 2, 3, 4, 5,6, 7]))
        sel.class_range = np.uint8(np.arange(0, 10))
        #[nndb_tr, nndb_val, nndb_te, _, _, _, _] = DbSlice.slice(nndb, sel)

        # define a data workspace to work with disk database
        db_dir = os.path.join(data_folder, "disk_db")

        # Use nndb, iterators read from the memory
        list_dbparams = self.__inmem_dbparams(nndb, sel)

        # Use database at db_dir, write the processed data on to the disk, iterators read from the disk
        # list_dbparams = self.__indsk_dbparams(os.path.join(db_dir, "_processed_DB1"), sel)

        # Use nndb, write the processed data on to the disk, iterators read from the disk.
        #list_dbparams = self.__mem_to_dsk_indsk_dbparams(nndb, db_dir, sel)

        # Use nndb, write the processed data on to the disk, but iterators read from the memory.
        # list_dbparams = self.__mem_to_dsk_inmem_dbparams(nndb, db_dir, sel)

        nnpatchman = NNPatchMan(VGG16PatchGen(), list_dbparams)

        vggcfg = VGG16Cfg()
        #vggcfg.numepochs = 2
        vggcfg.weights_dir = model_folder
        nnpatchman.predict(vggcfg)

    ##########################################################################
    # Private Interface
    ##########################################################################
    def __inmem_dbparams(self, nndb, sel):
        """Use nndb, iterators read from the memory"""
        dbparam1 = {'alias': "DB1",
                'nndb': nndb, 'selection': sel,
                'iter_param': {'class_mode':'categorical',
                                'batch_size':100
                                },
                'iter_pp_param': {'rescale':1./255, 'nrm_vgg16':True},
                'iter_in_mem': True}

        return dbparam1

    def __indsk_dbparams(self, db_dir, sel):
        """Use database at db_dir, write the processed data on to the disk, iterators read from the disk."""
        dbparam1 = {'alias': "DB1",
                'db_dir': db_dir, 'selection': sel,
                'iter_param': {'class_mode':'categorical',
                                'batch_size':100,
                                'target_size':(256,256)
                                },
                'iter_pp_param': {'rescale':1./255},
                'iter_in_mem': False}

        return dbparam1

    def __mem_to_dsk_indsk_dbparams(self, nndb, db_dir, sel):
        """Use nndb, write the processed data on to the disk, iterators read from the disk."""
        dbparam1 = {'alias': "DB1",
                'nndb': nndb, 'db_dir': db_dir, 'selection': sel, 
                'iter_param': {'class_mode':'categorical',
                                'batch_size':100,
                                'target_size':(256,256)
                                },
                'iter_pp_param': {'rescale':1./255},
                'iter_in_mem': False}

        return dbparam1

    def __mem_to_dsk_inmem_dbparams(self, nndb, db_dir, sel):
        """Use nndb, write the processed data on to the disk, iterators read from the memory."""
        dbparam1 = {'alias': "DB1",
                'nndb': nndb, 'db_dir': db_dir, 'selection': sel, 
                'iter_param': {'class_mode':'categorical',
                                'batch_size':100
                                },
                'iter_pp_param': {'rescale':1./255},
                'iter_in_mem': True}
        
        return dbparam1

    ##########################################################################
    # NNModel: Callbacks
    ##########################################################################
    @staticmethod
    def _fn_predict(nnmodel, nnpatch, predictions, true_output):
        #pass

        nndb_fe = NNdb('features', predictions[1], 8, True, cls_lbl=None, format=Format.N_H)
        sel_fe = Selection()
        sel_fe.use_rgb = False
        sel_fe.tr_col_indices      = np.array([0, 2, 3, 5], dtype='uint8')
        sel_fe.te_col_indices      = np.array([6, 7], dtype='uint8')
        sel_fe.class_range         = np.uint8(np.arange(0, 10))
        [nndb_tr_fe, _, nndb_te_fe, _, _, _, _] = DbSlice.slice(nndb_fe, sel_fe)

        W, info = LDA.l2(nndb_tr_fe)
        accuracy = Util.test(W, nndb_tr_fe, nndb_te_fe, info)
        print("LDA: Accuracy:" + str(accuracy))