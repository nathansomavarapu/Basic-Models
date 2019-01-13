import numpy as np
import cv2
import glob
import os

def gen_dset(dataset_path):
    labs = glob.glob(os.path.join(dataset_path, '*'))
    dset_paths = []

    for lab in labs:
        l_name = int(lab.split('/')[-1])
        for img in glob.glob(os.path.join(lab, '*.png')):
            dset_paths.append((img, l_name))

    return dset_paths

def sig(x):
  return 1 / (1 + np.exp(-x))

def sig_normalized(x, theta):
    u_probs = sig(np.matmul(theta.T, x))
    probs = u_probs/np.sum(u_probs)
    return np.reshape(probs, (1, probs.shape[0]))

def load_and_linearize(pth, intercept):
    img = cv2.imread(pth, cv2.IMREAD_GRAYSCALE)
    m,n = img.shape
    img = np.reshape(img, m*n)

    if intercept:
        tmp = np.ones(m*n + 1)
        tmp[:img.shape[0]] = img
        img = tmp
    
    return img

def fit_binary_logistic_regression(dset, data_len, intercept, iterations=10000, alpha=0.001):
    theta = np.random.random(data_len)

    rand_data = dset[np.random.choice(len(dset), size=iterations)]

    for d in rand_data:
        img, lab = d
        img = load_and_linearize(img, intercept)
        lab = int(lab)
        theta += alpha * (lab - sig(np.dot(theta, img))) * img
    
    return theta

def fit_softmax_regression(dset, data_len, intercept, num_cl, iterations=10000, alpha=0.001):
    theta = np.random.random((data_len, num_cl))

    rand_data = dset[np.random.choice(len(dset), size=iterations)]

    for d in rand_data:
        img, lab = d
        img = load_and_linearize(img, intercept)
        img = np.reshape(img, (img.shape[0], 1))
        lab = int(lab)

        inds = np.zeros((1, num_cl), dtype=np.float)
        inds[:, lab] = 1.0
        inds = inds - sig_normalized(img, theta)

        theta += alpha * np.matmul(img, inds)
    
    return theta
    

if __name__ == "__main__":
    dataset_path = '../data/mnist_png/training/'
    testset_path = '../data/mnist_png/testing/'

    dset = gen_dset(dataset_path)
    testset = gen_dset(testset_path)

    intercept = True

    num_classes = 10

    dset_0_1 = np.array(list(filter(lambda x: (x[1] == 0 or x[1] == 1), dset)))

    original_img_size = cv2.imread(dset_0_1[0][0], cv2.IMREAD_GRAYSCALE).shape
    d0 = load_and_linearize(dset_0_1[0][0], intercept)

    theta_lr = fit_binary_logistic_regression(dset_0_1, d0.shape[0], intercept)

    viz_size = d0.shape[0] -1 if intercept else d0.shape[0]

    weight_img = np.reshape(theta_lr[:viz_size], original_img_size)
    cv2.imwrite('lr_theta_img.png', weight_img * 128)

    test_set_0_1 = np.array(list(filter(lambda x: (x[1] == 0 or x[1] == 1), testset)))

    test_imgs = np.zeros((test_set_0_1.shape[0], d0.shape[0]))
    for i in range(test_set_0_1.shape[0]):
        test_imgs[i,:] = load_and_linearize(test_set_0_1[i][0], intercept)
    
    test_labels = test_set_0_1[:,1].astype(np.int)

    preds = sig(np.matmul(test_imgs, theta_lr)).astype(np.int)
    print('Logistic Regression')
    print('Percent Incorrect on Test Set: ' + str(100 * np.sum((preds != test_labels).astype(np.int))/test_imgs.shape[0]) + '%')

    dset = np.array(dset)

    original_img_size = cv2.imread(dset[0][0], cv2.IMREAD_GRAYSCALE).shape
    d0 = load_and_linearize(dset[0][0], intercept)

    theta_softmax = fit_softmax_regression(dset, d0.shape[0], intercept, num_classes)

    viz_size = d0.shape[0] -1 if intercept else d0.shape[0]

    for i in range(num_classes):
        weight_img = np.reshape(theta_softmax[:,i][:viz_size], original_img_size)
        cv2.imwrite('theta_softmax_img_' + str(i) + '.png', weight_img * 128)
    
    test_set = np.array(testset)

    test_imgs = np.zeros((test_set.shape[0], d0.shape[0]))
    for i in range(test_set.shape[0]):
        test_imgs[i,:] = load_and_linearize(test_set[i][0], intercept)
    
    test_labels = test_set[:,1].astype(np.int)

    # preds = sig(np.matmul(test_imgs, theta_softmax))
    # print('Logistic Regression')
    # print('Percent Incorrect on Test Set: ' + str(100 * np.sum((preds != test_labels).astype(np.int))/test_imgs.shape[0]) + '%')



    
