import os
import shlex
import argparse
from tqdm import tqdm
import pickle as cPickle
import multiprocessing
import gzip
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from parmap import parmap
from sklearn.decomposition import PCA
from functools import partial

def parseArgs(parser):
    parser.add_argument('--labels_test', 
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train', 
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix',
                        default='_SIFT_patch_pr.pkl.gz',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--in_test',
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float, 
                        help='C parameter of the SVM')
    parser.add_argument('--compute-esvm', action='store_true',
                      help='Compute E-SVM (even with GMP)')
    parser.add_argument('--custom-descs', action='store_true',
                    help='Use custom SIFT descriptors')
    parser.add_argument('--multi-vlad', action='store_true',
                        help='Use multi-VLAD with PCA')
    parser.add_argument('--n-codebooks', type=int, default=5,
                        help='Number of codebooks for multi-VLAD')
    return parser

def getFiles(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    # get filenames from labelfile
    all_files = []
    labels = []
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if str(file_name).endswith(p):
                file_name = file_name.replace(p,'')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels



def loadRandomDescriptors(files, max_descriptors):
    # let's just take 100 files to speed-up the process
    max_files = 100
    indices = np.random.permutation(max_files)
    selected_files = np.random.choice(files, min(len(files), max_files), replace=False)
   
    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []

    for file_path in tqdm(selected_files, desc='Loading descriptors'):
        try:
            # handling both image files and pre-computed descriptors
            if str(file_path).endswith(('.png', '.jpg', '.jpeg')):
                desc = computeDescs(file_path)
            else:
                with gzip.open(file_path, 'rb') as f:
                    desc = cPickle.load(f, encoding='latin1')
            
            if desc is None or len(desc) == 0:
                print(f"Warning: No descriptors found in {file_path}")
                continue
                
            if len(desc) > max_descs_per_file:
                indices = np.random.choice(len(desc), max_descs_per_file, replace=False)
                desc = desc[indices]
                
            descriptors.append(desc)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

    if not descriptors:
        raise ValueError("No valid descriptors found in any files!")
    
    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors

def dictionary(descriptors, n_clusters):
    """ 
    return cluster centers for the descriptors 
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    kmeans.fit(descriptors)

    # Return the cluster centroids
    return kmeans.cluster_centers_

def assignments(descriptors, clusters):
    """ 
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix  470776x64
        clusters: KxD cluster matrix 100x64
    returns: TxK assignment matrix
    """
    distances = np.linalg.norm(descriptors[:, np.newaxis,:] - clusters[np.newaxis, : , :], axis=2) 
    # compute the distance between each descriptor and each cluster center
    #  shape (T, K), where T is the number of descriptors and K is the number of clusters

    nearest_cluster = np.argmin(distances, axis=1) # axis=1 : operate along the rows
    # create hard assignment
    assignment = np.zeros( (len(descriptors), len(clusters)) )
    assignment[np.arange(len(descriptors)), nearest_cluster] = 1 
    # sets the element at row i and column nearest_cluster[i] to 1 
    # and the rest of the elements in the row to 0

    # a = np.zeros((descriptors.shape[0], clusters.shape[0]), dtype=np.float32)
    # bf = cv2.BFMatcher(cv2.NORM_L2)
    # nearestCentroidIndex = bf.knnMatch(descriptors, clusters, k=1) 
    # for i, m in enumerate(nearestCentroidIndex):
    #     a[i, m[0].trainIdx] = 1 # the index of the nearest cluster center is set to 1

    return assignment


def vlad(files, mus, powernorm, gmp=False, gamma=1):
    K = mus.shape[0] # number of clusters
    encodings = [] # list of encodingsa
    
    for f in tqdm(files):
        if f.endswith(('.png', '.jpg', '.jpeg', '.tiff')): # two different icdar17 datasets, png -> binary images and jpg -> color images. Our test/train labels are of the binary images
            desc = computeDescs(f)
        else:
            with gzip.open(f, 'rb') as ff:
                desc = cPickle.load(ff, encoding='latin1')
        
        if desc is None or len(desc) == 0:
            f_enc = np.zeros(K * mus.shape[1], dtype=np.float32)
        else:
            a = assignments(desc, mus) # a: TxK
        
            T,D = desc.shape
            f_enc = np.zeros((K, D), dtype=np.float32) 
            for k in range(K):
                j = np.where(a[:, k] == 1)[0]
                if len(j) > 0:  # Ensure there are descriptors assigned
                    residuals = desc[j] - mus[k]  # Compute residuals for cluster k, 
                    # residuals mean the difference between the descriptor and the cluster center
                    if gmp: 
                        # max pooling
                        clf = Ridge(alpha=gamma, solver='sparse_cg', 
                                  fit_intercept=False, max_iter=500)
                        clf.fit(residuals, np.ones(len(residuals)))
                        f_enc[k, :] = clf.coef_  # GMP residuals
                    else:
                        f_enc[k, :] = np.sum(residuals, axis=0)  # Aggregate residuals
            
            f_enc = f_enc.flatten()

        if powernorm:
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))  # Power normalization

        f_enc /= np.linalg.norm(f_enc) + 1e-12 # L2 normalization 

        encodings.append(f_enc)

    encodings = np.vstack(encodings)   #shape: NxK*D  = (3600, 6400)

    return normalize(encodings, norm='l2')


def esvm(encs_test, encs_train, C=1000):
    """ 
    compute a new embedding using Exemplar Classification
    compute for each encs_test encoding an E-SVM using the
    encs_train as negatives   
    parameters: 
        encs_test: NxD matrix
        encs_train: MxD matrix

    returns: new encs_test matrix (NxD)
    """

    # set up labels
    # TODO
    args = [(i, encs_test, encs_train, C) for i in range(len(encs_test))]


    def loop(arg):
        i, encs_test, encs_train, C = arg
        # i is the index of the test descriptor
        # encs_test is the test encoding matrix
        # encs_train is the training encoding matrix
        # C is the regularization parameter - this is the parameter that controls the trade-off between the margin and the classification error
        
        test_descriptor = encs_test[i]
        x = np.vstack([encs_train, test_descriptor[np.newaxis, :]])  # stack vertically and add a new dimension
        y = np.hstack([np.full(len(encs_train), -1), 1])  # creating labels for training data

        clf = LinearSVC(C=C, class_weight="balanced", max_iter=10000)  # Initialize a linear SVM
        clf.fit(x, y)

        x_new = normalize(clf.coef_, norm='l2').flatten()  # normalize the coefficients and flatten the array
        return x_new 

    # Parallelize the computation
    new_encs = list(map(loop, tqdm(args)))
    new_encs = np.stack(new_encs)
    print(new_encs.shape)

    return new_encs


def distances(encs):
    """ 
    compute pairwise distances 

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """ 
    similarity = np.dot(encs, encs.T)  # Compute dot product between encodings
    dists = 1 - similarity  # Compute cosine distance
    np.fill_diagonal(dists, np.finfo(dists.dtype).max) # mask out distance with itself

    return dists

def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    dist_matrix = distances(encs)
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[ indices[r,k] ] == labels[ r ]:
                rel += 1
                precisions.append( rel / float(k+1) )
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)
    top1 = float(correct) / n_encs

    print('Top-1 accuracy: {} - mAP: {}'.format(top1, mAP))
    return top1, mAP


def computeDescs(fileName: str) -> np.ndarray:
    # print(f"Processing image: {fileName}")
    img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image {fileName}") # error handling (was needed because got confused between the two icdar17 datasets)
        return np.array([])
    
    # using opencv sift
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)

    for kp in keypoints:
        kp.angle = 0
    
    keypoints, descriptors = sift.compute(img, keypoints)
    if descriptors is None:
        return np.array([])
    
    if descriptors.shape[1] == 128:
        descriptors = descriptors[:, ::2]
    
    l1_norms = np.linalg.norm(descriptors, ord=1, axis=1, keepdims=True)
    l1_norms[l1_norms == 0] = 1  # avoid division by zero
    descriptors = descriptors / l1_norms
    descriptors = np.sign(descriptors) * np.sqrt(np.abs(descriptors))
    
    if descriptors is None or len(descriptors) == 0:
        print(f"Warning: No descriptors found in {fileName}")
        return np.array([])

    # if descriptors is not None and len(descriptors) > 0:
    #     save_path = fileName.replace('.png', '_SIFT_custom.pkl.gz')
    #     with gzip.open(save_path, 'wb') as f:
    #         cPickle.dump(descriptors, f)
    
    return descriptors

def multivlad(files, all_mus, powernorm, gmp=False, gamma=1):
    encodings = []
    for mus in all_mus: # for each codebook
        # VLAD encodings
        vlad_enc = vlad(files, mus, powernorm, gmp, gamma)
        encodings.append(vlad_enc)
    
    return np.hstack(encodings)


#------------------------------------------------------------------------------------------------
def main():
    # ======================== Argument Parsing ========================
    parser = argparse.ArgumentParser('retrieval')
    parser.add_argument('--use-images', action='store_true',
                      help='Use original images instead of pre-computed descriptors')
    parser.add_argument('--output-dir', type=str,
                      default=r'C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx3\data\outputs',
                      help='Base output directory')
    parser.add_argument('--evaluate-all', action='store_true',
                      help='Run all possible combinations of methods')
    parser = parseArgs(parser)
    args = parser.parse_args()

    # ======================== Path Configuration =======================
    # Base data paths remain the same
    local_feat_path = r'C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx3\data\icdar17_local_features\icdar17_local_features'
    original_images_path_test = r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx3\data\ScriptNet-HistoricalWI-2017-binarized"
    original_images_path_train = r"C:\Users\taimo\Desktop\computer-vision-project\CV-projectEx3\data\icdar17-historicalwi-training-binarized\icdar2017-training-binary"
    if args.use_images:
        args.in_train = original_images_path_train
        args.in_test = original_images_path_test
        args.suffix_train = '.png'
        args.suffix_test = '.jpg'
    else:
        args.in_train = os.path.join(local_feat_path, 'train')
        args.in_test = os.path.join(local_feat_path, 'test')

    args.labels_train = os.path.join(local_feat_path, 'icdar17_labels_train.txt')
    args.labels_test = os.path.join(local_feat_path, 'icdar17_labels_test.txt')

    # Validate paths and create directories
    for path in [args.labels_train, args.labels_test]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    dirs = {
        'output': args.output_dir,
        'codebook': os.path.join(args.output_dir, 'codebooks'),
        'encoding': os.path.join(args.output_dir, 'encodings'),
        'results': os.path.join(args.output_dir, 'results')
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # ======================== Data Loading ============================
    print("Loading data files...")
    files_train, labels_train = getFiles(args.in_train, args.suffix_train, args.labels_train)
    files_test, labels_test = getFiles(args.in_test, args.suffix_test, args.labels_test)
    print(f'Loaded {len(files_train)} training and {len(files_test)} test files')

    # ====================== Task Execution ===========================
    results = {}  # Store all evaluation results

    # Basic Codebook Generation
    print("\n=== Task A: Basic Codebook Generation ===")
    single_codebook_path = os.path.join(dirs['codebook'], 'single_codebook.pkl.gz')
    if not os.path.exists(single_codebook_path) or args.overwrite:
        print("Generating single codebook...")
        descriptors = loadRandomDescriptors(files_train, 500000)
        mus = dictionary(descriptors, 100)
        with gzip.open(single_codebook_path, 'wb') as f:
            cPickle.dump(mus, f)
    else:
        with gzip.open(single_codebook_path, 'rb') as f:
            mus = cPickle.load(f)

    # Basic VLAD
    print("\n=== Task B: Basic VLAD Encoding ===")
    enc_train_basic = vlad(files_train, mus, powernorm=False, gmp=False)
    enc_test_basic = vlad(files_test, mus, powernorm=False, gmp=False)
    results['basic_vlad'] = evaluate(enc_test_basic, labels_test)

    # VLAD with Power Normalization
    print("\n=== Task C: VLAD with Power Normalization ===")
    enc_train_power = vlad(files_train, mus, powernorm=True, gmp=False)
    enc_test_power = vlad(files_test, mus, powernorm=True, gmp=False)
    results['power_norm_vlad'] = evaluate(enc_test_power, labels_test)

    # Exemplar Classification
    print("\n=== Task D: Exemplar Classification ===")
    enc_test_esvm = esvm(enc_test_power, enc_train_power, C=1000)
    results['esvm'] = evaluate(enc_test_esvm, labels_test)

    if args.evaluate_all:

        # SIFT features
        if args.use_images:
            print("\n=== Task E: Custom SIFT Features ===")
            results['custom_sift'] = results['power_norm_vlad']

        # =Task F: GMP
        print("\n=== Generalized Max Pooling ===")
        for gamma in [0.5]: #, 1, 2, 5, 10]:
            print(f"\nTesting GMP with gamma={gamma}")

            print("without E-SVM")
            # Without E-SVM
            enc_train_gmp = vlad(files_train, mus, powernorm=True, gmp=True, gamma=gamma)
            enc_test_gmp = vlad(files_test, mus, powernorm=True, gmp=True, gamma=gamma)
            results[f'gmp_gamma_{gamma}'] = evaluate(enc_test_gmp, labels_test)

            print("with E-SVM")
            # With E-SVM
            enc_test_gmp_esvm = esvm(enc_test_gmp, enc_train_gmp, C=1000)
            results[f'gmp_esvm_gamma_{gamma}'] = evaluate(enc_test_gmp_esvm, labels_test)

        # Task G: Multi-VLAD with PCA
        print("\n=== Multi-VLAD with PCA ===")
        # Generate multiple codebooks
        multi_codebooks = []
        for seed in range(5):
            descriptors = loadRandomDescriptors(files_train, 500000)
            kmeans = MiniBatchKMeans(n_clusters=32, random_state=seed)
            kmeans.fit(descriptors)
            multi_codebooks.append(kmeans.cluster_centers_)

        # Compute concatenated VLAD encodings
        enc_train_multi = multivlad(files_train, multi_codebooks, powernorm=True)
        
        # Apply PCA
        pca = PCA(n_components=1000, whiten=True)
        enc_train_multi = pca.fit_transform(enc_train_multi)
        
        # Transform test encodings
        enc_test_multi = multivlad(files_test, multi_codebooks, powernorm=True)
        enc_test_multi = pca.transform(enc_test_multi)
        
        results['multi_vlad_pca'] = evaluate(enc_test_multi, labels_test)

        # Multi-VLAD + E-SVM
        enc_test_multi_esvm = esvm(enc_test_multi, enc_train_multi, C=1000)
        results['multi_vlad_pca_esvm'] = evaluate(enc_test_multi_esvm, labels_test)

    # ====================== Results Summary ==========================
    print("\n=== Final Results Summary ===")
    print("\nMethod\t\t\t\tTop-1 Accuracy\tmAP")
    print("-" * 50)
    for method, (acc, map_score) in results.items():
        print(f"{method:<30}\t{acc:.4f}\t{map_score:.4f}")

    # Save results
    results_path = os.path.join(dirs['results'], 'evaluation_results.pkl.gz')
    with gzip.open(results_path, 'wb') as f:
        cPickle.dump(results, f)
    print(f"\nResults saved to {results_path}")

if __name__ == '__main__':
    main()


# python exercise3.py --evaluate_all --overwrite
# python exercise3.py --use-images --evaluate-all --overwrite
# python exercise3.py 