# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 03:08:06 2020

@author: Ashraf
"""


# importing the libraries
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential


import numpy as np
from numpy.random import seed
from random import seed
rand = 205
np.random.seed(rand)
seed(9001)


class Classify:
    def ROC_Val(matrix):
        '''
        Calculating the fales positive, false negative, true positive, true negative 
        plotting the ROC value of the result
        '''

        TP=matrix[0][0]
        FP=matrix[1][0]
        TN=matrix[1][1]
        FN=matrix[0][1]
        N = TN+FP
        P = TP+FN
        tp_rates = TP/P
        fp_rates = FP/N
        ROC_values = tp_rates/fp_rates

        print('Total = ' , P+N)
        print( matrix,'\n')

        print('hams  P = ' , P)
        print('TP = ' , TP)
        print('FN = ' , FN)
        print('tp_rate tp/p : ' , tp_rates,'\n')

        print('Spams N = ' , N)
        print('TN = ' , TN)
        print('FP = ' , FP)
        print('fp_rate fp/N : ' , fp_rates,'\n')
        
        print('ROC_value : ' , ROC_values,'\n')
        
        fprx = fp_rates
        tpry = tp_rates
        
        import matplotlib.pyplot as plt
        plt.plot(fprx, tpry, '-x' ,lw=1, label='ROC_Curve ' )
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve and AUC for SOM')
        plt.legend(loc="lower right")
        
        return ROC_values, fprx, tpry


    def words_win_cel(mappings, vocabulary):
        '''
            getting the winning cel address(x,y) for each vocabulary
        '''
        keys = list(mappings.keys())
        values = list(mappings.values())
        map_cels = (np.zeros(len(vocabulary))).tolist()
        
        for i, cel in enumerate(keys):
            voc_indx = values[i]
            for indx in voc_indx:
                map_cels[indx] = cel
        return map_cels


    def create_dens_img(review_vocab, map_cels, vocabulary, not_foundWords, X_of_map, y_of_map):
        '''
            getting the review images based on cell density with removing all the not covered vocabularies in glove dictionary
        '''
        #reviews = review_vocab
        reviews_dens_img = []
        for rev in review_vocab:
            one_rev_img = np.zeros((X_of_map, y_of_map))
            for r in rev:
                if r not in not_foundWords:
                    indx = vocabulary.index(r)
                    #print(map_cels[indx])
                    cel = map_cels[indx]
                    x_cel = cel[0]
                    y_cel = cel[1]
                    one_rev_img[x_cel][y_cel] = one_rev_img[x_cel][y_cel] + 1
            reviews_dens_img.append(one_rev_img)   
        return reviews_dens_img


    def create_freq_img(review_vocab, map_cels, vocabulary, missed_tfidfW, not_foundWords, X_of_map, y_of_map, df_TF_IDF):
        '''
            getting the review images based on word frequency with removing all the not covered vocabularies in glove dictionary
        '''
        missed_words = missed_tfidfW + not_foundWords
        reviews_img_freq= []
        for i, rev in enumerate(review_vocab):
            one_rev_img = np.zeros((X_of_map, y_of_map))
            for rr in rev:
                if rr not in missed_words:
                    indx = vocabulary.index(rr)
                    cel = map_cels[indx]
                    #freq = word_freq[indx]
                    x_cel = cel[0]
                    y_cel = cel[1]
                    freq = df_TF_IDF.at[i, rr]
                    one_rev_img[x_cel][y_cel] = one_rev_img[x_cel][y_cel] + freq
            reviews_img_freq.append(one_rev_img)
        return  reviews_img_freq


    def create_model(g, X_train, y_train, X_test, y_test):
    
        model = Sequential()
        #32 filter with the size of 3
        model.add(Convolution2D(32, 3, data_format ='channels_last', activation='relu', input_shape= (g,g,2)))#number of filters, the kernel size, and the activation function
        model.add(MaxPooling2D(pool_size= (2,2)))
        # Flattend nodes
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dropout(0.5))
        model.add(Dense(2))
        model.add(Activation('sigmoid'))
        model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        #model.summary()
        #monitor = EarlyStopping(monitor = 'val_loss', min_delta = 1e-3, patience = 10, verbose =1, mode = 'auto',
        #                      restore_best_weights = True)
        monitor = EarlyStopping(monitor = 'val_loss', min_delta = 1e-3, patience = 10, verbose =1, mode = 'auto')
                                
        model.fit(X_train, y_train, validation_data =(X_test, y_test), callbacks = [monitor], 
                verbose = 0, epochs =1000 , batch_size = 20)
        
        return model


    def reshape_img(reviews_dens_img, reviews_img_freq):    

        # Reshaping the images 
        dens_img = reviews_dens_img
        dens_img =np.array(dens_img)
        rev_dens_img = dens_img/dens_img.max()
        rev_dens_img = rev_dens_img[..., np.newaxis]

        rev_img_freq = reviews_img_freq
        rev_img_freq=np.array(rev_img_freq)
        rev_img_freq = rev_img_freq/rev_img_freq.max()
        rev_img_freq=rev_img_freq[..., np.newaxis]

        # combining X_img and rev_img_freq
        review_images = np.concatenate((rev_dens_img, rev_img_freq), axis=3)

        return review_images

