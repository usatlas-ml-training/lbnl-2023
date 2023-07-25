import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

def reweight(events,model,batch_size=10000):
    f = model.predict(events, batch_size=batch_size)
    weights = f / (1. - f)
    return np.squeeze(np.nan_to_num(weights))

# Binary crossentropy for classifying two samples with weights
# Weights are "hidden" by zipping in y_true (the labels)

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    
    return K.mean(t_loss)

def dis_preprocessing(events,mean=None,std=None):
    "input features are obs_hfs_pt", "obs_hfs_eta", "obs_e_e", "obs_e_pz", "obs_e_eta, let's do a simple standardization"
    new_data = np.copy(events)
    new_data[:,0] = np.log(new_data[:,0])
    new_data[:,2] = np.log(new_data[:,2])
    
    if mean is None:
        mean = np.mean(new_data,0,keepdims=True)
        std = np.std(new_data,0,keepdims=True)        

    return (new_data-mean)/std,mean,std

def omnifold(theta0_G,theta0_S,theta_unknown_S,iterations,model,verbose=0):

    if any(len(lst) != len(theta0_S) for lst in [theta0_S, theta0_G, theta_unknown_S]):
        print("all inputs must be of same length")
        exit

    weights = np.empty(shape=(iterations, 2, len(theta0_S)))
    # shape = (iteration, step, event)
    
    labels0 = np.zeros(len(theta0_S)) #synthetic theta0_S label = 0
    labels_unknown = np.ones(len(theta_unknown_S)) #data, theta_unknown_S
    
    xvals_1 = np.concatenate((theta0_S, theta_unknown_S))
    yvals_1 = np.concatenate((labels0, labels_unknown))

    xvals_2 = np.concatenate((theta0_G, theta0_G))
    yvals_2 = np.concatenate((labels0, labels_unknown))

    # initial iterative weights are ones
    weights_pull = np.ones(len(theta0_S))
    weights_push = np.ones(len(theta0_S))
    
    for i in range(iterations):

        print("\nITERATION: {}\n".format(i + 1))
        
        
        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data

        
        print("STEP 1\n")
        pass
            
        weights_1 = np.concatenate((weights_push, np.ones(len(theta_unknown_S))))

        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(xvals_1, yvals_1, weights_1)

        # zip ("hide") the weights with the labels
        Y_train_1 = np.stack((Y_train_1, w_train_1), axis=1)
        Y_test_1 = np.stack((Y_test_1, w_test_1), axis=1)   
        
        model.compile(loss=weighted_binary_crossentropy,
                      optimizer='Adam',
                      metrics=['accuracy'])

        model.fit(X_train_1,
                  Y_train_1,
                  epochs=40,
                  batch_size=1000,
                  validation_data=(X_test_1, Y_test_1),
                  verbose=verbose)

        weights_pull = weights_push * reweight(theta0_S,model)
        weights[i, 0, :] = weights_pull

        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        
        print("\nSTEP 2\n")
        pass

        weights_2 = np.concatenate((np.ones(len(theta0_G)), weights_pull))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.

        X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(xvals_2, yvals_2, weights_2)

        # zip ("hide") the weights with the labels
        Y_train_2 = np.stack((Y_train_2, w_train_2), axis=1)
        Y_test_2 = np.stack((Y_test_2, w_test_2), axis=1)   
        
        model.compile(loss=weighted_binary_crossentropy,
                      optimizer='Adam',
                      metrics=['accuracy'])
        model.fit(X_train_2,
                  Y_train_2,
                  epochs=40,
                  batch_size=1000,
                  validation_data=(X_test_2, Y_test_2),
                  verbose=verbose)
        
        weights_push = reweight(theta0_G,model)
        weights[i, 1, :] = weights_push
        pass
        
    return weights
